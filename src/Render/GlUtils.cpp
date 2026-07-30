#include "GlUtils.h"
#include "Logger.h"

#include "GL/glew.h"

bool checkGlError(const char* label)
{
	bool bFoundError= false;

	GLenum errorCode= glGetError();
	while (errorCode != GL_NO_ERROR)
	{
		const char* errorString= "UNKNOWN_GL_ERROR";
		switch (errorCode)
		{
		case GL_INVALID_ENUM:
			errorString= "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			errorString= "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			errorString= "GL_INVALID_OPERATION";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			errorString= "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			errorString= "GL_OUT_OF_MEMORY";
			break;
		}

		MIKAN_LOG_ERROR("checkGlError") << label << " - " << errorString << " (0x" << std::hex << errorCode << std::dec
										<< ")";

		bFoundError= true;
		errorCode= glGetError();
	}

	return bFoundError;
}
