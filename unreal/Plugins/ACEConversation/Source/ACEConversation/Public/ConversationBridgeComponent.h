#pragma once

#include "Components/ActorComponent.h"
#include "ConversationBridgeComponent.generated.h"

class IWebSocket;

UENUM(BlueprintType)
enum class EACEConversationState : uint8
{
    Listening,
    Thinking,
    Speaking
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FACEStateChanged, EACEConversationState, NewState);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FACETranscriptDelta, const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FACELlmDelta, const FString&, Text);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FACETtsStarted, int32, SampleRateHz);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FACETtsEnded);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_ThreeParams(
    FACETtsAudioChunk,
    const TArray<uint8>&,
    Pcm16Mono,
    int32,
    SampleRateHz,
    const FGuid&,
    TurnId
);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FACEConversationError, const FString&, Code, const FString&, Message);

UCLASS(ClassGroup=(ACE), meta=(BlueprintSpawnableComponent))
class ACECONVERSATION_API UConversationBridgeComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UConversationBridgeComponent();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="ACE")
    FString ServerUrl = TEXT("ws://127.0.0.1:8080/ws/session");

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACEStateChanged OnStateChanged;

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACETranscriptDelta OnAsrPartial;

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACETranscriptDelta OnAsrFinal;

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACELlmDelta OnLlmDelta;

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACETtsStarted OnTtsStarted;

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACETtsAudioChunk OnTtsAudioChunk;

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACETtsEnded OnTtsEnded;

    UPROPERTY(BlueprintAssignable, Category="ACE")
    FACEConversationError OnError;

    UFUNCTION(BlueprintCallable, Category="ACE")
    void Connect();

    UFUNCTION(BlueprintCallable, Category="ACE")
    void Disconnect();

    UFUNCTION(BlueprintCallable, Category="ACE")
    void PushMicChunk(const TArray<uint8>& Pcm16Mono);

    UFUNCTION(BlueprintCallable, Category="ACE")
    void SendMicEnd();

private:
    TSharedPtr<IWebSocket> Socket;
    FGuid SessionId;
    FGuid ActiveTurnId;
    bool bSessionStarted = false;

    void HandleConnected();
    void HandleConnectionError(const FString& ErrorMessage);
    void HandleClosed(int32 StatusCode, const FString& Reason, bool bWasClean);
    void HandleTextMessage(const FString& Message);
    void HandleBinaryMessage(const void* Data, SIZE_T Size, SIZE_T BytesRemaining);

    void SendSessionStart();
    TArray<uint8> BuildMicAudioFrame(const TArray<uint8>& Pcm16Mono) const;
    static bool ParseTtsAudioFrame(const void* Data, SIZE_T Size, int32& OutSampleRateHz, TArray<uint8>& OutPayload);
    static void AppendUint32BE(TArray<uint8>& Buffer, uint32 Value);
    static uint32 ReadUint32BE(const uint8* Bytes);
};
