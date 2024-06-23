#version 120

// Input from vertex shader
varying vec2 texCoord;

// The texture sampler
uniform sampler2D albedoMap;
uniform sampler2D emissionMap;

void main() {
    vec3 albedo = texture2D(albedoMap, texCoord).rgb;
    vec3 emission = texture2D(emissionMap, texCoord).rgb;

    vec3 color = albedo + emission*2;
    gl_FragColor = vec4(color, 1.0);
}