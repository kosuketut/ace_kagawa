using UnrealBuildTool;

public class ACEAvatarSandbox : ModuleRules
{
    public ACEAvatarSandbox(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "InputCore",
                "EnhancedInput",
                "ACEConversation"
            }
        );
    }
}

