Shader "ProceduralSceneGate/MetricDepthUint24Millimetres"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual
            CGPROGRAM
            #pragma target 3.0
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct v2f
            {
                float4 position : SV_POSITION;
                float view_depth_m : TEXCOORD0;
            };

            v2f vert(appdata_base input)
            {
                v2f output;
                output.position = UnityObjectToClipPos(input.vertex);
                output.view_depth_m = max(0.0, -UnityObjectToViewPos(input.vertex).z);
                return output;
            }

            float4 frag(v2f input) : SV_Target
            {
                // Exact metric storage in a lossless RGB PNG. The raw integer is
                // millimetres, little endian: mm = R + 256*G + 65536*B.
                float millimetres = clamp(floor(input.view_depth_m * 1000.0 + 0.5), 1.0, 16777215.0);
                float red = fmod(millimetres, 256.0);
                float green = fmod(floor(millimetres / 256.0), 256.0);
                float blue = floor(millimetres / 65536.0);
                return float4(float3(red, green, blue) / 255.0, 1.0);
            }
            ENDCG
        }
    }
    Fallback Off
}
