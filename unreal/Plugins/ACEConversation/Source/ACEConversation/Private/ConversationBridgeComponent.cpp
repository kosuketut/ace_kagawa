#include "ConversationBridgeComponent.h"

#include "Dom/JsonObject.h"
#include "IWebSocket.h"
#include "Modules/ModuleManager.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "WebSocketsModule.h"

namespace
{
static constexpr ANSICHAR AceMagic[4] = {'A', 'C', 'E', '1'};
static constexpr uint8 FrameVersion = 1;
static constexpr uint8 MicKind = 1;
static constexpr uint8 TtsKind = 2;
static constexpr uint8 PcmS16Le = 1;
static constexpr int32 HeaderSize = 32;
}

UConversationBridgeComponent::UConversationBridgeComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UConversationBridgeComponent::Connect()
{
    if (Socket.IsValid() && Socket->IsConnected())
    {
        return;
    }

    FWebSocketsModule& WebSocketsModule = FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
    Socket = WebSocketsModule.CreateWebSocket(ServerUrl);

    Socket->OnConnected().AddUObject(this, &UConversationBridgeComponent::HandleConnected);
    Socket->OnConnectionError().AddUObject(this, &UConversationBridgeComponent::HandleConnectionError);
    Socket->OnClosed().AddUObject(this, &UConversationBridgeComponent::HandleClosed);
    Socket->OnMessage().AddUObject(this, &UConversationBridgeComponent::HandleTextMessage);
    Socket->OnRawMessage().AddUObject(this, &UConversationBridgeComponent::HandleBinaryMessage);
    Socket->Connect();
}

void UConversationBridgeComponent::Disconnect()
{
    bSessionStarted = false;
    ActiveTurnId.Invalidate();
    if (Socket.IsValid())
    {
        Socket->Close();
        Socket.Reset();
    }
}

void UConversationBridgeComponent::PushMicChunk(const TArray<uint8>& Pcm16Mono)
{
    if (!Socket.IsValid() || !Socket->IsConnected() || !bSessionStarted || Pcm16Mono.IsEmpty())
    {
        return;
    }
    const TArray<uint8> Frame = BuildMicAudioFrame(Pcm16Mono);
    Socket->Send(Frame.GetData(), Frame.Num(), true);
}

void UConversationBridgeComponent::SendMicEnd()
{
    if (!Socket.IsValid() || !Socket->IsConnected() || !bSessionStarted)
    {
        return;
    }

    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("type"), TEXT("mic.end"));
    Root->SetStringField(TEXT("session_id"), SessionId.ToString(EGuidFormats::DigitsWithHyphensLower));
    Root->SetObjectField(TEXT("payload"), MakeShared<FJsonObject>());

    FString Json;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Json);
    FJsonSerializer::Serialize(Root, Writer);
    Socket->Send(Json);
}

void UConversationBridgeComponent::HandleConnected()
{
    SessionId = FGuid::NewGuid();
    bSessionStarted = true;
    SendSessionStart();
}

void UConversationBridgeComponent::HandleConnectionError(const FString& ErrorMessage)
{
    OnError.Broadcast(TEXT("connection_error"), ErrorMessage);
}

void UConversationBridgeComponent::HandleClosed(int32 StatusCode, const FString& Reason, bool bWasClean)
{
    (void)StatusCode;
    (void)Reason;
    (void)bWasClean;
    bSessionStarted = false;
    ActiveTurnId.Invalidate();
    OnStateChanged.Broadcast(EACEConversationState::Listening);
}

void UConversationBridgeComponent::HandleTextMessage(const FString& Message)
{
    TSharedPtr<FJsonObject> Root;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
    {
        return;
    }

    FString Type;
    if (!Root->TryGetStringField(TEXT("type"), Type))
    {
        return;
    }
    const TSharedPtr<FJsonObject>* Payload = nullptr;
    Root->TryGetObjectField(TEXT("payload"), Payload);

    FString TurnIdString;
    Root->TryGetStringField(TEXT("turn_id"), TurnIdString);
    if (!TurnIdString.IsEmpty())
    {
        FGuid::Parse(TurnIdString, ActiveTurnId);
    }

    if (Type == TEXT("state") && Payload && Payload->IsValid())
    {
        const FString StateString = (*Payload)->GetStringField(TEXT("state"));
        if (StateString == TEXT("LISTENING"))
        {
            OnStateChanged.Broadcast(EACEConversationState::Listening);
        }
        else if (StateString == TEXT("THINKING"))
        {
            OnStateChanged.Broadcast(EACEConversationState::Thinking);
        }
        else if (StateString == TEXT("SPEAKING"))
        {
            OnStateChanged.Broadcast(EACEConversationState::Speaking);
        }
        return;
    }

    if ((Type == TEXT("asr.partial") || Type == TEXT("asr.final") || Type == TEXT("llm.delta")) && Payload && Payload->IsValid())
    {
        const FString Text = (*Payload)->GetStringField(TEXT("text"));
        if (Type == TEXT("asr.partial"))
        {
            OnAsrPartial.Broadcast(Text);
        }
        else if (Type == TEXT("asr.final"))
        {
            OnAsrFinal.Broadcast(Text);
        }
        else
        {
            OnLlmDelta.Broadcast(Text);
        }
        return;
    }

    if (Type == TEXT("tts.start") && Payload && Payload->IsValid())
    {
        OnTtsStarted.Broadcast(static_cast<int32>((*Payload)->GetNumberField(TEXT("sample_rate_hz"))));
        return;
    }

    if (Type == TEXT("tts.end"))
    {
        OnTtsEnded.Broadcast();
        return;
    }

    if (Type == TEXT("error") && Payload && Payload->IsValid())
    {
        OnError.Broadcast((*Payload)->GetStringField(TEXT("code")), (*Payload)->GetStringField(TEXT("message")));
    }
}

void UConversationBridgeComponent::HandleBinaryMessage(const void* Data, SIZE_T Size, SIZE_T BytesRemaining)
{
    (void)BytesRemaining;
    int32 SampleRateHz = 0;
    TArray<uint8> Payload;
    if (!ParseTtsAudioFrame(Data, Size, SampleRateHz, Payload))
    {
        return;
    }

    OnTtsAudioChunk.Broadcast(Payload, SampleRateHz, ActiveTurnId);
}

void UConversationBridgeComponent::SendSessionStart()
{
    if (!Socket.IsValid() || !Socket->IsConnected())
    {
        return;
    }

    TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetStringField(TEXT("type"), TEXT("session.start"));
    Root->SetStringField(TEXT("session_id"), SessionId.ToString(EGuidFormats::DigitsWithHyphensLower));

    TSharedRef<FJsonObject> Payload = MakeShared<FJsonObject>();
    Payload->SetStringField(TEXT("locale"), TEXT("ja-JP"));
    Root->SetObjectField(TEXT("payload"), Payload);

    FString Json;
    const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Json);
    FJsonSerializer::Serialize(Root, Writer);
    Socket->Send(Json);
}

TArray<uint8> UConversationBridgeComponent::BuildMicAudioFrame(const TArray<uint8>& Pcm16Mono) const
{
    TArray<uint8> Buffer;
    Buffer.Reserve(HeaderSize + Pcm16Mono.Num());

    Buffer.Append(reinterpret_cast<const uint8*>(AceMagic), 4);
    Buffer.Add(FrameVersion);
    Buffer.Add(MicKind);
    Buffer.Add(PcmS16Le);
    Buffer.Add(1);
    AppendUint32BE(Buffer, 16000);
    AppendUint32BE(Buffer, static_cast<uint32>(Pcm16Mono.Num()));
    Buffer.AddZeroed(16);
    Buffer.Append(Pcm16Mono);
    return Buffer;
}

bool UConversationBridgeComponent::ParseTtsAudioFrame(const void* Data, SIZE_T Size, int32& OutSampleRateHz, TArray<uint8>& OutPayload)
{
    if (Size < HeaderSize)
    {
        return false;
    }

    const uint8* Bytes = static_cast<const uint8*>(Data);
    if (FMemory::Memcmp(Bytes, AceMagic, 4) != 0)
    {
        return false;
    }
    if (Bytes[4] != FrameVersion || Bytes[5] != TtsKind || Bytes[6] != PcmS16Le || Bytes[7] != 1)
    {
        return false;
    }

    const uint32 SampleRate = ReadUint32BE(Bytes + 8);
    const uint32 PayloadSize = ReadUint32BE(Bytes + 12);
    if (HeaderSize + PayloadSize > Size)
    {
        return false;
    }

    OutSampleRateHz = static_cast<int32>(SampleRate);
    OutPayload.SetNumUninitialized(static_cast<int32>(PayloadSize));
    FMemory::Memcpy(OutPayload.GetData(), Bytes + HeaderSize, PayloadSize);
    return true;
}

void UConversationBridgeComponent::AppendUint32BE(TArray<uint8>& Buffer, uint32 Value)
{
    Buffer.Add(static_cast<uint8>((Value >> 24) & 0xFF));
    Buffer.Add(static_cast<uint8>((Value >> 16) & 0xFF));
    Buffer.Add(static_cast<uint8>((Value >> 8) & 0xFF));
    Buffer.Add(static_cast<uint8>(Value & 0xFF));
}

uint32 UConversationBridgeComponent::ReadUint32BE(const uint8* Bytes)
{
    return (static_cast<uint32>(Bytes[0]) << 24)
        | (static_cast<uint32>(Bytes[1]) << 16)
        | (static_cast<uint32>(Bytes[2]) << 8)
        | static_cast<uint32>(Bytes[3]);
}
