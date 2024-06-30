#version 120

attribute vec4 p3d_Vertex;
varying vec2 texCoord;

void main() {
    gl_Position = p3d_Vertex;
}