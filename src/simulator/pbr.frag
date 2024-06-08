#version 120

varying vec3 fragPos;
varying vec3 normal;
varying vec2 texCoord;
varying vec4 fragPosLightSpace;

uniform sampler2D albedoMap;
uniform sampler2D emissionMap;
uniform sampler2D specularMap;
uniform sampler2D shadowMap;
uniform sampler2D cloudMap;

uniform vec3 lightPos;
uniform vec3 viewPos;

const float ambientStrength = 0.0;
const float specularStrength = 0.0;
const float shininess = 128.0;

uniform vec2 shadowMapSize;



float calculateShadow(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    float currentDepth = projCoords.y;
    vec2 texelSize = 1.0 / shadowMapSize;
    float shadow = 0.0;

    for(int x = -1; x <= 1; x++) {
        for(int y = -1; y <= 1; y++) {
            float sampleDepth = texture2D(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r;
            float shadowIntensity = mix(0.5, 0.01, clamp((currentDepth - sampleDepth) / 0.08, 0.0, 1.0));
            shadow += shadowIntensity;
        }
    }
    shadow /= 4.0;
    return shadow;
}

void main() {
    vec3 albedo = texture2D(albedoMap, texCoord).rgb;
    vec3 emission = texture2D(emissionMap, texCoord).rgb;
    vec3 specular = texture2D(specularMap, texCoord).rgb;
    float cloud = texture2D(cloudMap, texCoord).r;

    // Lighting
    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(lightPos - fragPos);
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    
    // Ambient
    vec3 ambient = ambientStrength * albedo;

    // Diffuse
    float diff = max(dot(norm, lightDir), 1.0);
    vec3 diffuse = diff * albedo * (1-specular*0.5);

    // Specular
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specularColor = specularStrength * spec * specular;

    // Shadow calculation
    float shadow = calculateShadow(fragPosLightSpace);


    vec3 result = (diffuse+specularColor+cloud) * shadow + max(emission*(1-shadow)-cloud*2,0);


    
    gl_FragColor = vec4(result, 1.0);
}
