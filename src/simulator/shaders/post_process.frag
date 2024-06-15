#version 120

uniform sampler2D tex;
uniform vec2 texel_size;
varying vec2 texcoord;
uniform float diagramValue;


void main() {

    // If the diagram value is 0, then we don't want to draw the outline
    if (diagramValue == 0) {
        gl_FragColor = texture2D(tex, texcoord);
    } else {
    

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

    }
}