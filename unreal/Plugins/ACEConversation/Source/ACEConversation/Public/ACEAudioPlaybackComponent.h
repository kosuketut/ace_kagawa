#pragma once

#include "Components/ActorComponent.h"
#include "ACEAudioPlaybackComponent.generated.h"

class UAudioComponent;
class USoundWaveProcedural;

UCLASS(ClassGroup=(ACE), meta=(BlueprintSpawnableComponent))
class ACECONVERSATION_API UACEAudioPlaybackComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UACEAudioPlaybackComponent();

    UFUNCTION(BlueprintCallable, Category="ACE")
    void StartPlayback(int32 SampleRateHz);

    UFUNCTION(BlueprintCallable, Category="ACE")
    void PushPcm16(const TArray<uint8>& Pcm16Mono, int32 SampleRateHz);

    UFUNCTION(BlueprintCallable, Category="ACE")
    void EndPlayback();

    UFUNCTION(BlueprintCallable, Category="ACE")
    void StopPlayback();

private:
    UPROPERTY(Transient)
    TObjectPtr<USoundWaveProcedural> ProceduralWave;

    UPROPERTY(Transient)
    TObjectPtr<UAudioComponent> AudioComponent;

    int32 CurrentSampleRateHz = 0;

    void EnsurePlaybackObjects(int32 SampleRateHz);
};

