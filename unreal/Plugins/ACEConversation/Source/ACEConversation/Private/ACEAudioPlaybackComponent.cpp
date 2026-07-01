#include "ACEAudioPlaybackComponent.h"

#include "Components/AudioComponent.h"
#include "GameFramework/Actor.h"
#include "Sound/SoundWaveProcedural.h"

UACEAudioPlaybackComponent::UACEAudioPlaybackComponent()
{
    PrimaryComponentTick.bCanEverTick = false;
}

void UACEAudioPlaybackComponent::StartPlayback(int32 SampleRateHz)
{
    EnsurePlaybackObjects(SampleRateHz);
    if (AudioComponent != nullptr && !AudioComponent->IsPlaying())
    {
        AudioComponent->Play();
    }
}

void UACEAudioPlaybackComponent::PushPcm16(const TArray<uint8>& Pcm16Mono, int32 SampleRateHz)
{
    if (Pcm16Mono.IsEmpty())
    {
        return;
    }

    EnsurePlaybackObjects(SampleRateHz);
    if (ProceduralWave != nullptr)
    {
        ProceduralWave->QueueAudio(Pcm16Mono.GetData(), Pcm16Mono.Num());
    }
    if (AudioComponent != nullptr && !AudioComponent->IsPlaying())
    {
        AudioComponent->Play();
    }
}

void UACEAudioPlaybackComponent::EndPlayback()
{
    if (AudioComponent != nullptr && AudioComponent->IsPlaying())
    {
        AudioComponent->FadeOut(0.05f, 0.0f);
    }
}

void UACEAudioPlaybackComponent::StopPlayback()
{
    if (AudioComponent != nullptr)
    {
        AudioComponent->Stop();
    }
}

void UACEAudioPlaybackComponent::EnsurePlaybackObjects(int32 SampleRateHz)
{
    if (AudioComponent != nullptr && ProceduralWave != nullptr && CurrentSampleRateHz == SampleRateHz)
    {
        return;
    }

    CurrentSampleRateHz = SampleRateHz;

    if (AudioComponent == nullptr)
    {
        AActor* Owner = GetOwner();
        if (Owner == nullptr)
        {
            return;
        }
        AudioComponent = NewObject<UAudioComponent>(Owner);
        AudioComponent->bAutoActivate = false;
        AudioComponent->bIsUISound = false;
        AudioComponent->RegisterComponent();
        Owner->AddInstanceComponent(AudioComponent);
    }

    ProceduralWave = NewObject<USoundWaveProcedural>(this);
    ProceduralWave->SetSampleRate(SampleRateHz);
    ProceduralWave->NumChannels = 1;
    ProceduralWave->Duration = INDEFINITELY_LOOPING_DURATION;
    ProceduralWave->bLooping = false;
    AudioComponent->SetSound(ProceduralWave);
}
