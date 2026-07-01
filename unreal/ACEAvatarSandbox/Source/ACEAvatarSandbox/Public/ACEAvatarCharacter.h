#pragma once

#include "GameFramework/Character.h"
#include "ACEAvatarCharacter.generated.h"

class UACEAudioPlaybackComponent;
class UConversationBridgeComponent;
class USceneComponent;

UCLASS()
class ACEAVATARSANDBOX_API AACEAvatarCharacter : public ACharacter
{
    GENERATED_BODY()

public:
    AACEAvatarCharacter();

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="ACE")
    TObjectPtr<USceneComponent> AvatarRoot;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="ACE")
    TObjectPtr<UConversationBridgeComponent> ConversationBridge;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="ACE")
    TObjectPtr<UACEAudioPlaybackComponent> AudioPlayback;

protected:
    virtual void BeginPlay() override;

private:
    UFUNCTION()
    void HandleTtsStarted(int32 SampleRateHz);

    UFUNCTION()
    void HandleTtsChunk(const TArray<uint8>& Pcm16Mono, int32 SampleRateHz, const FGuid& TurnId);

    UFUNCTION()
    void HandleTtsEnded();
};

