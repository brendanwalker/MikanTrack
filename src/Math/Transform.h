#pragma once

#include "MikanMathExport.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/gtx/matrix_decompose.hpp>

class GlmTransform
{
public:
	GlmTransform()
		: m_position(glm::vec3(0.f))
		, m_rotation(glm::quat())
		, m_scale(glm::vec3(1.f, 1.f, 1.f))
		, m_mat(glm::mat4(1.f))
	{
	}

	GlmTransform(const glm::vec3& position)
		: m_position(position)
		, m_rotation(glm::quat())
		, m_scale(glm::vec3(1.f, 1.f, 1.f))
	{
		rebuildMat();
	}

	GlmTransform(const glm::vec3& position, const glm::quat& rotation)
		: m_position(position)
		, m_rotation(rotation)
		, m_scale(glm::vec3(1.f, 1.f, 1.f))
	{
		rebuildMat();
	}

	GlmTransform(const glm::vec3& position, const glm::quat& rotation, const glm::vec3& scale)
		: m_position(position)
		, m_rotation(rotation)
		, m_scale(scale)
	{
		rebuildMat();
	}

	GlmTransform(const glm::mat4& mat4)
		: m_position(glm::vec3(0.f))
		, m_rotation(glm::quat())
		, m_scale(glm::vec3(1.f, 1.f, 1.f))
	{
		setMat4(mat4);
	}

	const glm::vec3& getScale() const { return m_scale; }

	const glm::quat& getRotation() const { return m_rotation; }

	const glm::vec3& getPosition() const { return m_position; }

	const glm::mat4& getMat4() const { return m_mat; }

	void setScale(const glm::vec3& scale)
	{
		m_scale= scale;
		rebuildMat();
	}

	void setRotation(const glm::quat& orientation)
	{
		m_rotation= orientation;
		rebuildMat();
	}

	void setPosition(const glm::vec3& position)
	{
		m_position= position;
		rebuildMat();
	}

	void setTransform(const glm::vec3& position, const glm::quat& rotation, const glm::vec3& scale)
	{
		m_position= position;
		m_rotation= rotation;
		m_scale= scale;
		rebuildMat();
	}

	void setMat4(const glm::mat4& mat4)
	{
		// Extract position scale and rotation from the local transform
		glm::vec3 scale;
		glm::quat orientation;
		glm::vec3 translation;
		glm::vec3 skew;
		glm::vec4 perspective;
		if (glm::decompose(mat4, scale, orientation, translation, skew, perspective))
		{
			m_position= translation;
			m_rotation= orientation;
			m_scale= scale;

			// Since the matrix could have had skew or perspective transforms in it
			// we have to rebuild it from just the decomposed TRS values
			rebuildMat();
		}
	}

	void appendScale(const glm::vec3& deltaScale)
	{
		m_scale+= deltaScale;
		rebuildMat();
	}

	void appendRotation(const glm::quat& deltaRotation)
	{
		m_rotation= deltaRotation * m_rotation;
		rebuildMat();
	}

	void appendTranslation(const glm::vec3& deltaPosition)
	{
		m_position+= deltaPosition;
		rebuildMat();
	}

private:
	void rebuildMat()
	{
		const glm::mat4 scale= glm::scale(glm::mat4(1.f), m_scale);
		const glm::mat4 rotation= glm::mat4_cast(m_rotation);
		const glm::mat4 translation= glm::translate(glm::mat4(1.f), m_position);

		m_mat= translation * (rotation * scale);
	}

	glm::vec3 m_position;
	glm::quat m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_mat;
};
