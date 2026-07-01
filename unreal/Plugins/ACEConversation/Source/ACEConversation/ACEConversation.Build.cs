using UnrealBuildTool;

public class ACEConversation : ModuleRules
{
    public ACEConversation(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new[]
            {
                "Core",
                "CoreUObject",
                "Engine"
            }
        );

        PrivateDependencyModuleNames.AddRange(
            new[]
            {
                "Json",
                "Projects",
                "WebSockets"
            }
        );
    }
}
