/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include "GLFW\glfw3.h"
float up_rot=0;
float right_rot =0 ;
glm::vec2 all_amt = glm::vec2(0,0);//translate along x,y
float posx1;
float posy1;
float posx;
float posy;

bool click = false;
float previous_t;
bool rotation = false;
bool translate = false;

//-------------------------------
//-------------MAIN--------------
//-------------------------------
int main(int argc, char **argv) {
	if (argc != 2) {
		cout << "Usage: [obj file]" << endl;

		return 0;
	}

	obj *mesh = new obj();

	{
		objLoader loader(argv[1], mesh);
		mesh->buildBufPoss();
	}

	frame = 0;
	seconds = time(NULL);
	fpstracker = 0;

	// Launch CUDA/GL
	if (init(mesh)) {

		// GLFW main loop
		mainLoop();
	}
	getchar();
	return 0;
}
//mouse function

void mainLoop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		
		runCuda();

		time_t seconds2 = time(NULL);

		if (seconds2 - seconds >= 1) {

			fps = fpstracker / (seconds2 - seconds);
			fpstracker = 0;
			seconds = seconds2;
		}

		string title = "CIS565 Rasterizer | " + utilityCore::convertIntToString((int)fps) + " FPS";
		glfwSetWindowTitle(window, title.c_str());

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glBindTexture(GL_TEXTURE_2D, displayImage);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glClear(GL_COLOR_BUFFER_BIT);

		// VAO, shader program, and texture already bound
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, 0);
		glfwSwapBuffers(window);
	}
	glfwDestroyWindow(window);
	glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------
//the camera movement refers to CIS-560 based code..  MyGL::keyPressEvent(QKeyEvent *e){..}
void mouse_pos_callback(GLFWwindow *window, double x_current, double y_current)
{
	float amount = 0.03;
	float deltaTime = glfwGetTime() -previous_t;

	if (rotation)//right
	{
		right_rot += amount * deltaTime * float(x_current - posx);
		up_rot += amount * deltaTime * float(y_current - posy);
	}
	else if (translate)//left
	{
		all_amt.x = amount * deltaTime * float(x_current - posx);
		all_amt.y = amount * deltaTime * float(y_current - posy);
 	}
	previous_t = glfwGetTime();
}
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	click = true;
	double xpos = 0, ypos = 0; double xpos1 = 0, ypos1 = 0;
	//double xpos, ypos; double xpos1, ypos1;
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		translate = true;//press left button move..
		rotation = false;//glfwGetCursorPos(window, &xpos, &ypos);...not a good idea to put inside...
	}
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		/*glfwGetCursorPos(window, &xpos1, &ypos1);
		if (xpos1 - xpos > 0){all_amt += 1;}
		if (xpos1 - xpos <= 0){all_amt += -1;}*/
		translate = false;
		rotation = false;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
		translate = false;//press right button rotate..
		rotation = true;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
	{
		translate = false;
		rotation = false;
	}
}
void mouse_scroll_callback(GLFWwindow* window,double front,double back)
{
	//......
}
void runCuda() {
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	dptr = NULL;
	//Camera camera;
	/*
	glfwSetCursorPosCallback(window, mouse_move_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, mouse_scroll_callback);*/
	//if (!click){
	//	all_amt=0;
	//}
	//glm::mat4 projview = camera.PerspectiveProjectionMatrix;
	cudaGLMapBufferObject((void **)&dptr, pbo);

	rasterize(dptr, all_amt.x, all_amt.x,up_rot, right_rot);
	cudaGLUnmapBufferObject(pbo);

	frame++;
	fpstracker++;
	click=false;

}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(obj *mesh) {
	glfwSetErrorCallback(errorCallback);

	if (!glfwInit()) {
		return false;
	}

	width = 800;
	height = 800;
	window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glfwSetCursorPosCallback(window, mouse_pos_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	//glfwSetScrollCallback(window, mouse_scroll_callback);
	glfwSetKeyCallback(window, keyCallback);

	// Set up GL context
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		return false;
	}

	// Initialize other stuff
	initVAO();
	initTextures();
	initCuda();
	initPBO();

	float cbo[] = {
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		1.0, 0.0, 0.0
	};
	rasterizeSetBuffers(mesh->getBufIdxsize(), mesh->getBufIdx(),
		mesh->getBufPossize() / 3,
		mesh->getBufPos(), mesh->getBufNor(), mesh->getBufCol(),1);

	GLuint passthroughProgram;
	passthroughProgram = initShader();

	glUseProgram(passthroughProgram);
	glActiveTexture(GL_TEXTURE0);

	return true;
}

void initPBO() {
	// set up vertex data parameter
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte)* num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);

}

void initCuda() {
	// Use device with highest Gflops/s
	cudaGLSetGLDevice(0);

	rasterizeInit(width, height);

	// Clean up on program exit
	atexit(cleanupCuda);
}

void initTextures() {
	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
		GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
		-1.0f, 1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() {
	const char *attribLocations[] = { "Position", "Tex" };
	GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
	GLint location;

	glUseProgram(program);
	if ((location = glGetUniformLocation(program, "u_image")) != -1) {
		glUniform1i(location, 0);
	}

	return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda() {
	if (pbo) {
		deletePBO(&pbo);
	}
	if (displayImage) {
		deleteTexture(&displayImage);
	}
}

void deletePBO(GLuint *pbo) {
	if (pbo) {
		// unregister this buffer object with CUDA
		cudaGLUnregisterBufferObject(*pbo);

		glBindBuffer(GL_ARRAY_BUFFER, *pbo);
		glDeleteBuffers(1, pbo);

		*pbo = (GLuint)NULL;
	}
}


void deleteTexture(GLuint *tex) {
	glDeleteTextures(1, tex);
	*tex = (GLuint)NULL;
}

void shut_down(int return_code) {
	rasterizeFree();
	cudaDeviceReset();
#ifdef __APPLE__
	glfwTerminate();
#endif
	exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char *description) {
	fputs(description, stderr);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

