#version 120

attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;
attribute vec2 p3d_MultiTexCoord0;

varying vec3 fragPos;
varying vec3 normal;
varying vec2 texCoord;
varying vec4 fragPosLightSpace;

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 lightSpaceMatrix;

void main() {
    fragPos = vec3(p3d_ModelMatrix * p3d_Vertex);
    normal = normalize(p3d_NormalMatrix * p3d_Normal);
    texCoord = p3d_MultiTexCoord0;

    fragPosLightSpace = lightSpaceMatrix * p3d_ModelMatrix * p3d_Vertex;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}

// uniform sampler2D topographyMap; // Add topographic texture sampler

// void main() {
//     // Get texture coordinates
//     texCoord = p3d_MultiTexCoord0;

//     // Sample topographic texture
//     float topographicValue = texture2D(topographyMap, texCoord).r;

//     // Displace vertex position based on topographic value
//     vec3 displacedPos = p3d_Vertex.xyz + p3d_Normal * topographicValue * 0.05;

//     // Transform the displaced position to view space
//     fragPos = vec3(p3d_ModelMatrix * vec4(displacedPos, 1.0));
//     normal = normalize(p3d_NormalMatrix * p3d_Normal);

//     // Calculate fragment position in light space for shadow mapping
//     fragPosLightSpace = lightSpaceMatrix * p3d_ModelMatrix * vec4(displacedPos, 1.0);

//     // Calculate final vertex position
//     gl_Position = p3d_ModelViewProjectionMatrix * vec4(displacedPos, 1.0);
// }
