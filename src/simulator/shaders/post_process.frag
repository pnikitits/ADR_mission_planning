#version 120

uniform sampler2D tex;
uniform sampler2D depthTex;
uniform vec2 texel_size;
uniform float diagramValue;
uniform float atmosphereValue;
uniform vec2 screenSize;
uniform vec3 uCameraPosition;
uniform mat4 uInverseProjectionMatrix;
uniform mat4 uInverseViewMatrix;
uniform vec3 scatteringCoefficients;

varying vec2 texcoord;

const int numOpticalDepthPoints = 10;
const int numInScatteringPoints = 10;
const float densityFalloff = 2.0;
const vec3 dirToSun = vec3(0.0, -1.0, 0.0);

const float planetRadius = 0.7;
const float atmosphereRadius = 0.8;

vec3 planetCentre = vec3(0.0, 0.0, 0.0);



vec3 getRayDir(vec2 uv) {
    // Convert UV to NDC (Normalized Device Coordinates)
    vec4 ndc = vec4((uv - 0.5) * 2.0, 1.0, 1.0);
    vec4 clipSpacePos = uInverseProjectionMatrix * ndc;
    vec3 viewSpacePos = (clipSpacePos / clipSpacePos.w).xyz;
    vec3 worldSpacePos = (uInverseViewMatrix * vec4(viewSpacePos, 1.0)).xyz;
    return normalize(worldSpacePos - uCameraPosition);
}

vec2 raySphere(vec3 sphereCentre, float sphereRadius, vec3 rayOrigin, vec3 rayDirection) {
    vec3 offset = rayOrigin - sphereCentre;
    float a = dot(rayDirection, rayDirection);
    float b = 2.0 * dot(offset, rayDirection);
    float c = dot(offset, offset) - sphereRadius * sphereRadius;
    float d = b * b - 4.0 * a * c;

    if (d >= 0.0) {
        float s = sqrt(d);
        float dstToSphereNear = max(0.0, (-b - s) / (2.0 * a));
        float dstToSphereFar = (-b + s) / (2.0 * a);

        if (dstToSphereFar >= 0.0) {
            return vec2(dstToSphereNear, dstToSphereFar - dstToSphereNear);
        }
    }

    float maxFloat = 3.402823466e+38;
    return vec2(maxFloat, 0);
}


float densityAtPoint(vec3 densitySamplePoint) {
    float heightAboveSurface = length(densitySamplePoint - planetCentre) - planetRadius;
    float height01 = heightAboveSurface / (atmosphereRadius - planetRadius);
    float localDensity = exp(-height01 * densityFalloff) * (1.0 - height01);
    return localDensity;
}


float opticalDepth(vec3 rayOrigin , vec3 rayDir , float rayLength) {
    vec3 densitySamplePoint = rayOrigin;
    float stepSize = rayLength / (numOpticalDepthPoints - 1);
    float opticalDepth = 0.0;

    for (int i = 0; i < numOpticalDepthPoints; i++) {
        float localDensity = densityAtPoint(densitySamplePoint);
        opticalDepth += localDensity * stepSize;
        densitySamplePoint += rayDir * stepSize;
    }
    return opticalDepth;
}


vec3 calculateLight(vec3 rayOrigin , vec3 rayDir , float rayLength , vec3 originalColor) {
    vec3 inScatterPoint = rayOrigin;
    float stepSize = rayLength / (numInScatteringPoints - 1);
    vec3 inScatteredLight = vec3(0.0);
    float viewRayOpticalDepth = 0.0;

    for (int i = 0; i < numInScatteringPoints; i++) {
        float sunRayLength = raySphere(planetCentre , atmosphereRadius , inScatterPoint , dirToSun).y;
        float sunRayOpticalDepth = opticalDepth(inScatterPoint , dirToSun , sunRayLength);
        float viewRayOpticalDepth = opticalDepth(inScatterPoint , -rayDir , stepSize*i);
        vec3 transmittance = exp(-(sunRayOpticalDepth + viewRayOpticalDepth) * scatteringCoefficients);
        float localDensity = densityAtPoint(inScatterPoint);

        inScatteredLight += localDensity * transmittance * scatteringCoefficients * stepSize;
        inScatterPoint += rayDir * stepSize;
    }
    float originalColTransmittance = exp(-viewRayOpticalDepth);
    return originalColor * originalColTransmittance + inScatteredLight;
}


void main() {
    if (atmosphereValue == 1.0 && diagramValue == 0.0) {

        vec4 originalColor = texture2D(tex, texcoord);

        
        float sceneDepthNonLinear = texture2D(depthTex, texcoord).r;
        
        float uNear = 0.1;
        float uFar = 100.0;
        float sceneDepth = uNear / (uFar - sceneDepthNonLinear * (uFar - uNear));
        sceneDepth = sceneDepth * 100.0;



        vec3 rayOrigin = uCameraPosition;
        vec3 rayDir = getRayDir(texcoord);


        vec2 hitInfo = raySphere(planetCentre, atmosphereRadius, rayOrigin, rayDir);
        float dstToAtmosphere = hitInfo.x;
        float dstThroughAtmosphere = min(hitInfo.y , sceneDepth - dstToAtmosphere);

        if (dstThroughAtmosphere > 0) {
            const float epsilon = 0.0001;
            vec3 pointInAtmosphere = rayOrigin + rayDir * (dstToAtmosphere+epsilon);
            vec3 light = calculateLight(pointInAtmosphere, rayDir, dstThroughAtmosphere-epsilon*2.0 , originalColor.rgb);
            gl_FragColor = vec4(light , 1.0);
        } else {
            gl_FragColor = originalColor;
        }



    } else if (diagramValue == 1.0 && atmosphereValue == 0.0) {
        // Draw the outline (for diagram use)
        float edge_threshold = 0.1;

        vec3 color = texture2D(tex, texcoord).rgb;
        
        vec3 color_up = texture2D(tex, texcoord + vec2(0.0, texel_size.y)).rgb;
        vec3 color_down = texture2D(tex, texcoord - vec2(0.0, texel_size.y)).rgb;
        vec3 color_left = texture2D(tex, texcoord - vec2(texel_size.x, 0.0)).rgb;
        vec3 color_right = texture2D(tex, texcoord + vec2(texel_size.x, 0.0)).rgb;
        
        float edge = length(color - color_up) + length(color - color_down) + length(color - color_left) + length(color - color_right);
        
        if (edge > edge_threshold) {
            gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);  // Outline color (black)
        } else {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 0.0);  // Transparent color
        }

    } else {
        gl_FragColor = texture2D(tex, texcoord);
    }
}