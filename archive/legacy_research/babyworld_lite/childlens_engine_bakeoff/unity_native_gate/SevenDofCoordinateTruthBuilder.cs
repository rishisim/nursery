using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

// Minimal bounded convention validator. Deliberately contains no skin, fingers,
// gravity, target, room, camera, collider, or contact code.
public static class SevenDofCoordinateTruthBuilder
{
    const float Dt = 1f / 240f;
    const float Epsilon = 1e-3f;
    static readonly string Output = Environment.GetEnvironmentVariable("SEVEN_DOF_OUTPUT");
    static readonly List<ArticulationBody> Bodies = new();
    static ArticulationBody root;
    static ArticulationBody palmBody;
    static Transform palmSite;

    static readonly Vector3[] Axes = {
        new(.31f,.91f,.27f), new(-.72f,.22f,.66f), new(.18f,-.64f,.75f),
        new(.09f,.96f,-.27f), new(-.58f,.51f,.64f), new(.77f,.19f,.61f),
        new(-.28f,.83f,.48f)
    };
    static readonly Vector3[] Links = {
        new(.055f,-.105f,.035f), new(.072f,-.082f,.041f), new(.061f,-.071f,.052f),
        new(.049f,-.058f,.043f), new(.037f,-.044f,.034f), new(.028f,-.032f,.027f),
        new(.024f,-.021f,.031f)
    };
    static readonly float[][] Poses = {
        new[]{.12f,-.26f,.19f,.63f,-.31f,.22f,-.17f},
        new[]{-.38f,.29f,.41f,.88f,-.46f,.36f,.21f},
        new[]{.44f,-.33f,.27f,.52f,.39f,-.28f,.34f}
    };

    [MenuItem("BabyWorld/Run Seven DOF Coordinate Truth Microgate")]
    public static void Run()
    {
        if (string.IsNullOrWhiteSpace(Output)) throw new Exception("SEVEN_DOF_OUTPUT is required");
        Directory.CreateDirectory(Output);
        var rows = new List<ColumnRow>();
        Build();
        var starts = new List<int>();
        int totalDofs = root.GetDofStartIndices(starts);
        var bodies = Bodies.Select(b => new BodyRow { name=b.name, index=b.index, dof_count=b.dofCount, dof_start=starts[b.index] }).ToArray();
        bool indexAudit = totalDofs == 7 && bodies.Length == 7;
        for (int i=0;i<bodies.Length;i++) indexAudit &= bodies[i].index == i + 1 && bodies[i].dof_count == 1 && bodies[i].dof_start == i;

        float roundtrip = 0f;
        foreach (var pose in Poses) {
            SetCoordinates(pose);
            var read = new List<float>(); root.GetJointPositions(read);
            for (int i=0;i<7;i++) roundtrip=Mathf.Max(roundtrip,Mathf.Abs(read[i]-pose[i]));
            var dense = new ArticulationJacobian((Bodies.Count + 2) * 6, totalDofs);
            root.GetDenseJacobian(ref dense);
            // Dense rows omit the immovable root while body.index includes it.
            int row = (palmBody.index - 1) * 6;
            Debug.Log($"SEVEN_DOF_DENSE rows={dense.rows} columns={dense.columns} palm_index={palmBody.index} row={row} total_dofs={totalDofs}");
            for (int column=0;column<7;column++) {
                var engineLinear = new Vector3(dense[row,column],dense[row+1,column],dense[row+2,column]);
                var engineAngular = new Vector3(dense[row+3,column],dense[row+4,column],dense[row+5,column]);
                PoseAt(Offset(pose,column,Epsilon),out var plusPosition,out var plusRotation);
                PoseAt(Offset(pose,column,-Epsilon),out var minusPosition,out var minusRotation);
                var finiteLinear=(plusPosition-minusPosition)/(2*Epsilon);
                var delta=plusRotation*Quaternion.Inverse(minusRotation); delta.ToAngleAxis(out float angleDeg,out Vector3 axis);
                if(angleDeg>180)angleDeg-=360;
                var finiteAngular=axis*(angleDeg*Mathf.Deg2Rad/(2*Epsilon));
                rows.Add(new ColumnRow { pose_rad=pose, column=column, body_index=Bodies[column].index,
                    dof_start=starts[Bodies[column].index], engine_linear=engineLinear, finite_linear=finiteLinear,
                    engine_angular=engineAngular, finite_angular=finiteAngular,
                    linear_direction_error_deg=Direction(engineLinear,finiteLinear), linear_relative_magnitude_error=Relative(engineLinear,finiteLinear),
                    angular_direction_error_deg=Direction(engineAngular,finiteAngular), angular_relative_magnitude_error=Relative(engineAngular,finiteAngular) });
                SetCoordinates(pose);
            }
        }
        var report=new Report { schema="embodied.seven_dof_coordinate_truth.v1",unity_version=Application.unityVersion,
            pose_count=Poses.Length,column_count=7,row_count=rows.Count,body_index_and_dof_start_indices_pass=indexAudit,
            coordinate_roundtrip_max_rad=roundtrip,
            max_linear_direction_error_deg=rows.Max(x=>x.linear_direction_error_deg),max_linear_relative_magnitude_error=rows.Max(x=>x.linear_relative_magnitude_error),
            max_angular_direction_error_deg=rows.Max(x=>x.angular_direction_error_deg),max_angular_relative_magnitude_error=rows.Max(x=>x.angular_relative_magnitude_error),bodies=bodies };
        report.passed=indexAudit&&roundtrip<=1e-6f&&report.max_linear_direction_error_deg<=2&&report.max_linear_relative_magnitude_error<=.03f&&report.max_angular_direction_error_deg<=2&&report.max_angular_relative_magnitude_error<=.03f;
        File.WriteAllText(Path.Combine(Output,"seven_dof_rows.json"),JsonUtility.ToJson(new RowList{rows=rows.ToArray()},true));
        File.WriteAllText(Path.Combine(Output,"seven_dof_report.json"),JsonUtility.ToJson(report,true));
        EditorApplication.Exit(report.passed?0:2);
    }

    static void Build()
    {
        foreach(var b in UnityEngine.Object.FindObjectsByType<ArticulationBody>(FindObjectsSortMode.None))UnityEngine.Object.DestroyImmediate(b.gameObject);
        Bodies.Clear();
        var go=new GameObject("seven_dof_root"); root=go.AddComponent<ArticulationBody>();root.immovable=true;root.useGravity=false;
        Transform parent=go.transform;
        for(int i=0;i<7;i++){
            var child=new GameObject($"joint_{i}");child.transform.SetParent(parent,false);child.transform.localPosition=i==0?Vector3.zero:Links[i-1];
            child.transform.localRotation=Quaternion.FromToRotation(Vector3.right,Axes[i].normalized);
            var body=child.AddComponent<ArticulationBody>();body.jointType=ArticulationJointType.RevoluteJoint;body.twistLock=ArticulationDofLock.LimitedMotion;body.useGravity=false;body.mass=.05f;
            var drive=body.xDrive;drive.lowerLimit=-120;drive.upperLimit=120;drive.stiffness=0;drive.damping=0;drive.forceLimit=0;body.xDrive=drive;
            Bodies.Add(body);parent=child.transform;
        }
        var site=new GameObject("palm_site");site.transform.SetParent(parent,false);site.transform.localPosition=Links[6];palmBody=site.AddComponent<ArticulationBody>();palmBody.jointType=ArticulationJointType.FixedJoint;palmBody.useGravity=false;palmBody.mass=.01f;palmSite=site.transform;
        Physics.simulationMode=SimulationMode.Script;Physics.gravity=Vector3.zero;Physics.SyncTransforms();Physics.Simulate(Dt);Physics.SyncTransforms();
    }
    static void SetCoordinates(float[] pose){root.SetJointPositions(pose.ToList());root.SetJointVelocities(Enumerable.Repeat(0f,7).ToList());Physics.SyncTransforms();Physics.Simulate(Dt);Physics.SyncTransforms();}
    static void PoseAt(float[] pose,out Vector3 position,out Quaternion rotation){SetCoordinates(pose);position=palmSite.position;rotation=palmSite.rotation;}
    static float[] Offset(float[] pose,int column,float delta){var result=(float[])pose.Clone();result[column]+=delta;return result;}
    static float Direction(Vector3 a,Vector3 b)=>(a.sqrMagnitude<1e-14f&&b.sqrMagnitude<1e-14f)?0:Vector3.Angle(a,b);
    static float Relative(Vector3 a,Vector3 b)=>Mathf.Abs(a.magnitude-b.magnitude)/Mathf.Max(1e-9f,Mathf.Max(a.magnitude,b.magnitude));

    [Serializable] class RowList{public ColumnRow[] rows;}
    [Serializable] class ColumnRow{public float[] pose_rad;public int column,body_index,dof_start;public Vector3 engine_linear,finite_linear,engine_angular,finite_angular;public float linear_direction_error_deg,linear_relative_magnitude_error,angular_direction_error_deg,angular_relative_magnitude_error;}
    [Serializable] class BodyRow{public string name;public int index,dof_count,dof_start;}
    [Serializable] class Report{public string schema,unity_version;public int pose_count,column_count,row_count;public bool body_index_and_dof_start_indices_pass,passed;public float coordinate_roundtrip_max_rad,max_linear_direction_error_deg,max_linear_relative_magnitude_error,max_angular_direction_error_deg,max_angular_relative_magnitude_error;public BodyRow[] bodies;}
}
