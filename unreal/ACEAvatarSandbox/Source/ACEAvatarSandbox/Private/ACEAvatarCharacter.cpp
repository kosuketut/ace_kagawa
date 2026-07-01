#include "ACEAvatarCharacter.h"

#include "ACEAudioPlaybackComponent.h"
#include "Components/SceneComponent.h"
#include "ConversationBridgeComponent.h"

AACEAvatarCharacter::AACEAvatarCharacter()
{
    AvatarRoot = CreateDefaultSubobject<USceneComponent>(TEXT("AvatarRoot"));
    AvatarRoot->SetupAttachment(GetRootComponent());

    ConversationBridge = CreateDefaultSubobject<UConversationBridgeComponent>(TEXT("ConversationBridge"));
    AudioPlayback = CreateDefaultSubobject<UACEAudioPlaybackComponent>(TEXT("AudioPlayback"));
}

void AACEAvatarCharacter::BeginPlay()
{
    Super::BeginPlay();

    if (ConversationBridge != nullptr && AudioPlayback != nullptr)
    {
        ConversationBridge->OnTtsStarted.AddDynamic(this, &AACEAvatarCharacter::HandleTtsStarted);
        ConversationBridge->OnTtsAudioChunk.AddDynamic(this, &AACEAvatarCharacter::HandleTtsChunk);
        ConversationBridge->OnTtsEnded.AddDynamic(this, &AACEAvatarCharacter::HandleTtsEnded);
    }
}

void AACEAvatarCharacter::HandleTtsStarted(int32 SampleRateHz)
{
    if (AudioPlayback != nullptr)
    {
        AudioPlayback->StartPlayback(SampleRateHz);
    }
}

void AACEAvatarCharacter::HandleTtsChunk(const TArray<uint8>& Pcm16Mono, int32 SampleRateHz, const FGuid& TurnId)
{
    (void)TurnId;
    if (AudioPlayback != nullptr)
    {
        AudioPlayback->PushPcm16(Pcm16Mono, SampleRateHz);
    }

    // When the NVIDIA ACE plugin is installed, forward the same PCM chunk to
    // FACERuntimeModule::Get().AnimateFromAudioSamples(...) using the
    // UACEAudioCurveSourceComponent attached to the MetaHuman.
}

void AACEAvatarCharacter::HandleTtsEnded()
{
    if (AudioPlayback != nullptr)
    {
        AudioPlayback->EndPlayback();
    }

    // When using raw runtime samples with Audio2Face-3D, also call
    // FACERuntimeModule::Get().EndAudioSamples(...) here.
}

