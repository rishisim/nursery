#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace EmbodiedReferenceHand
{
    public static class ReferenceHandMicrocellBuilder
    {
        [MenuItem("Embodied Simulation/Reference Hand Qualification/Build Player")]
        public static void BuildPlayer()
        {
            const string scenePath = "Assets/ReferenceHandQualification.unity";
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject bootstrap = new GameObject("ReferenceHandQualification");
            bootstrap.AddComponent<ReferenceHandQualification>();
            GameObject prefab = AssetDatabase.LoadAssetAtPath<GameObject>("Packages/com.ultraleap.tracking/Hands/Runtime/Prefabs/Built In Render Pipeline (Dynamically Upgradable)/GenericHand_Arm.prefab");
            if (prefab == null) throw new System.InvalidOperationException("GenericHand_Arm package prefab was not imported");
            GameObject visible = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            visible.name = "GenericHand_Arm_Visible";
            SceneManager.MoveGameObjectToScene(visible, bootstrap.scene);
            EditorSceneManager.SetActiveScene(bootstrap.scene);
            EditorSceneManager.SaveScene(bootstrap.scene, scenePath);
            BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = "../ReferenceHandQualification.app",
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.None
            });
            EditorApplication.Exit(report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded ? 0 : 2);
        }
    }
}
#endif
