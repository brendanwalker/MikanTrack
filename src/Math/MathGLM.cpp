//-- includes -----
#include "MathGLM.h"

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/common.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/euler_angles.hpp>

//-- public methods -----
bool glm_vec3_is_nearly_equal(const glm::vec3& a, const glm::vec3& b, const float epsilon)
{
	return is_nearly_equal(a.x, b.x, epsilon) && is_nearly_equal(a.y, b.y, epsilon)
		   && is_nearly_equal(a.z, b.z, epsilon);
}

float glm_vec3_normalize_with_default(glm::vec3& v, const glm::vec3& default_result)
{
	const float length= glm::length(v);

	// Use the default value if v is too tiny
	v= (length > k_normal_epsilon) ? (v / length) : default_result;

	return length;
}

glm::vec3 glm_vec3_lerp(const glm::vec3& a, const glm::vec3& b, const float u) { return a * (1.f - u) + b * u; }

glm::mat4 glm_scale_along_axis(const glm::vec3& axis, const float scale)
{
	const glm::vec3 unit_axis= glm::normalize(axis);

	// Rotate the axis to the x-axis
	const glm::quat undo_quat= glm::rotation(unit_axis, glm::vec3(1.f, 0.f, 0.f));
	const glm::mat4 undo_rot= glm::mat4_cast(undo_quat);

	// Scale along the x-axis by the scale factor
	const glm::mat4 x_scale= glm::scale(glm::mat4(1.f), glm::vec3(scale, 1.f, 1.f));

	// Re-apply the rotation back to what is was
	const glm::quat redo_quat= glm::inverse(undo_quat);
	const glm::mat4 redo_rot= glm::mat4_cast(redo_quat);

	// Combine all transforms together
	glm::mat scale_along_axis_xform= undo_rot;
	scale_along_axis_xform= glm_composite_xform(scale_along_axis_xform, x_scale);
	scale_along_axis_xform= glm_composite_xform(scale_along_axis_xform, redo_rot);

	return scale_along_axis_xform;
}

void glm_quat_to_euler_angles(const glm::quat& orientation, float& out_x_radians, float& out_y_radians,
							  float& out_z_radians)
{
	const glm::mat4 R= glm::mat4_cast(orientation);
	glm::extractEulerAngleXYZ(R, out_x_radians, out_y_radians, out_z_radians);
}

void glm_euler_angles_to_mat3(float x_radians, float y_radians, float z_radians, glm::mat3& out_orientation)
{
	out_orientation= glm::mat3(glm::eulerAngleXYZ(x_radians, y_radians, z_radians));
}

void glm_euler_angles_to_quat(float x_radians, float y_radians, float z_radians, glm::quat& out_orientation)
{
	glm::mat3 R;
	glm_euler_angles_to_mat3(x_radians, y_radians, z_radians, R);

	out_orientation= glm::quat_cast(R);
}

glm::mat4 glm_composite_xform(const glm::mat4& first, const glm::mat4& second)
{
	// http://www.c-jump.com/bcc/common/Talk3/Math/GLM/GLM.html#W01_0140_matrix_multiplication
	return second * first;
}

glm::quat glm_composite_rotation(const glm::quat& first, const glm::quat& second)
{
	// http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/#how-do-i-cumulate-2-rotations-
	return second * first;
}

glm::mat4 glm_relative_xform(const glm::mat4& parentWorldXform, const glm::mat4& childWorldXform)
{
	const glm::mat4 invParentXform= glm::inverse(parentWorldXform);

	return glm_composite_xform(childWorldXform, invParentXform);
}

glm::mat4 glm_mat4_from_pose(const glm::quat& orientation, const glm::vec3& position)
{
	glm::mat4 rot= glm::mat4_cast(orientation);
	glm::mat4 trans= glm::translate(glm::mat4(1.0f), position);
	glm::mat4 transform= trans * rot;

	return transform;
}

glm::vec3 glm_mat4_get_x_axis(const glm::mat4& xform)
{
	return xform[0]; // x-axis
}

glm::vec3 glm_mat4_get_y_axis(const glm::mat4& xform)
{
	return xform[1]; // y-axis
}

glm::vec3 glm_mat4_get_z_axis(const glm::mat4& xform)
{
	return xform[2]; // z-axis
}

glm::vec3 glm_mat4_get_position(const glm::mat4& xform)
{
	return xform[3]; // position
}

void glm_mat4_set_x_axis(glm::mat4& xform, const glm::vec3& v)
{
	xform[0]= glm::vec4(v, 0.f); // x-axis
}

void glm_mat4_set_y_axis(glm::mat4& xform, const glm::vec3& v)
{
	xform[1]= glm::vec4(v, 0.f); // y-axis
}

void glm_mat4_set_z_axis(glm::mat4& xform, const glm::vec3& v)
{
	xform[2]= glm::vec4(v, 0.f); // z-axis
}

void glm_mat4_set_rotation(glm::mat4& xform, const glm::mat3& R)
{
	glm_mat4_set_x_axis(xform, R[0]);
	glm_mat4_set_y_axis(xform, R[1]);
	glm_mat4_set_z_axis(xform, R[2]);
}

void glm_mat4_set_position(glm::mat4& xform, const glm::vec3& v)
{
	xform[3]= glm::vec4(v, 1.f); // position
}

void glm_xform_points(const glm::mat4& xform, glm::vec3* points, size_t point_count)
{
	for (size_t i= 0; i < point_count; i++)
	{
		points[i]= xform * glm::vec4(points[i], 1);
	}
}

void glm_xform_vectors(const glm::mat4& xform, glm::vec3* points, size_t point_count)
{
	for (size_t i= 0; i < point_count; i++)
	{
		points[i]= xform * glm::vec4(points[i], 0);
	}
}

bool glm_closest_point_on_ray_to_point(const glm::vec3& ray_start, const glm::vec3& ray_direction,
									   const glm::vec3& point, float& out_ray_closest_time,
									   glm::vec3& out_ray_closest_point)
{
	const glm::vec3 ray_start_to_point= point - ray_start;
	const glm::vec3 ray_unit_direction= glm::normalize(ray_direction);

	out_ray_closest_time= glm::dot(ray_start_to_point, ray_unit_direction);
	out_ray_closest_point= ray_start + ray_direction * out_ray_closest_time;

	return out_ray_closest_time >= 0.f;
}

bool glm_closest_point_on_ray_to_ray(const glm::vec3& ray1_start, const glm::vec3& ray1_direction,
									 const glm::vec3& ray2_start, const glm::vec3& ray2_direction,
									 float& out_ray1_closest_time, glm::vec3& out_ray1_closest_point)
{
	// see https://palitri.com/vault/stuff/maths/Rays%20closest%20point.pdf
	const glm::vec3& a= ray1_direction;
	const glm::vec3& b= ray2_direction;
	const glm::vec3& c= ray2_start - ray1_start;

	const float a_dot_a= glm::dot(a, a);
	const float a_dot_b= glm::dot(a, b);
	const float a_dot_c= glm::dot(a, c);
	const float b_dot_b= glm::dot(b, b);
	const float b_dot_c= glm::dot(b, c);

	const float denomenator= a_dot_a * b_dot_b - a_dot_b * a_dot_b;
	if (!is_nearly_zero(a_dot_a) &&   // i.e. ray1_direction != 0
		!is_nearly_zero(b_dot_b) &&   // i.e. ray2_direction != 0
		!is_nearly_zero(denomenator)) // i.e. rays not parallel
	{
		const float numerator= a_dot_c * b_dot_b - a_dot_b * b_dot_c;

		out_ray1_closest_time= numerator / denomenator;
		out_ray1_closest_point= ray1_start + ray1_direction * out_ray1_closest_time;
		return true;
	}
	else
	{
		out_ray1_closest_time= 0.f;
		out_ray1_closest_point= ray1_start;
		return false;
	}
}

bool glm_intersect_plane_with_ray(const glm::vec3& point_on_plane, const glm::vec3& plane_normal,
								  const glm::vec3& ray_start, const glm::vec3& ray_direction, float& outIntDistance,
								  glm::vec3& outIntPoint)
{
	const glm::vec3 ray_unit_direction= glm::normalize(ray_direction);

	if (glm::intersectRayPlane(ray_start, ray_unit_direction, point_on_plane, plane_normal, outIntDistance))
	{
		outIntPoint= ray_start + ray_unit_direction * outIntDistance;

		return true;
	}

	return false;
}

bool glm_intersect_tri_with_ray(const GlmTriangle& tri, const glm::vec3& ray_start, const glm::vec3& ray_direction,
								float& outIntDistance, glm::vec3& outIntPoint, glm::vec3& outIntNormal)
{
	const glm::vec3 ray_unit_direction= glm::normalize(ray_direction);

	glm::vec2 baryPosition;
	if (glm::intersectRayTriangle(ray_start, ray_unit_direction, tri.v0, tri.v1, tri.v2, baryPosition, outIntDistance))
	{
		const glm::vec3 edge1= tri.v1 - tri.v0;
		const glm::vec3 edge2= tri.v2 - tri.v0;

		outIntPoint= ray_start + ray_direction * outIntDistance;
		outIntNormal= glm::normalize(glm::cross(edge1, edge2));

		return true;
	}

	return false;
}

glm::vec3 glm_closest_point_on_triangle(const GlmTriangle& tri, const glm::vec3& p)
{
	// Barycentric / Voronoi-region method from Christer Ericson, "Real-Time Collision Detection" (section 5.1.5).
	const glm::vec3& a= tri.v0;
	const glm::vec3& b= tri.v1;
	const glm::vec3& c= tri.v2;

	const glm::vec3 ab= b - a;
	const glm::vec3 ac= c - a;
	const glm::vec3 ap= p - a;

	// Check if P is in the vertex region outside A
	const float d1= glm::dot(ab, ap);
	const float d2= glm::dot(ac, ap);
	if (d1 <= 0.f && d2 <= 0.f)
		return a; // barycentric (1, 0, 0)

	// Check if P is in the vertex region outside B
	const glm::vec3 bp= p - b;
	const float d3= glm::dot(ab, bp);
	const float d4= glm::dot(ac, bp);
	if (d3 >= 0.f && d4 <= d3)
		return b; // barycentric (0, 1, 0)

	// Check if P is in the edge region of AB, if so return projection of P onto AB
	const float vc= d1 * d4 - d3 * d2;
	if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
	{
		const float v= d1 / (d1 - d3);
		return a + v * ab; // barycentric (1-v, v, 0)
	}

	// Check if P is in the vertex region outside C
	const glm::vec3 cp= p - c;
	const float d5= glm::dot(ab, cp);
	const float d6= glm::dot(ac, cp);
	if (d6 >= 0.f && d5 <= d6)
		return c; // barycentric (0, 0, 1)

	// Check if P is in the edge region of AC, if so return projection of P onto AC
	const float vb= d5 * d2 - d1 * d6;
	if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
	{
		const float w= d2 / (d2 - d6);
		return a + w * ac; // barycentric (1-w, 0, w)
	}

	// Check if P is in the edge region of BC, if so return projection of P onto BC
	const float va= d3 * d6 - d5 * d4;
	if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
	{
		const float w= (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b); // barycentric (0, 1-w, w)
	}

	// P is inside the face region. Compute the closest point through its barycentric coordinates (u, v, w)
	const float denom= 1.f / (va + vb + vc);
	const float v= vb * denom;
	const float w= vc * denom;
	return a + ab * v + ac * w;
}

float glm_point_aabb_distance_sq(const glm::vec3& p, const glm::vec3& aabb_min, const glm::vec3& aabb_max)
{
	float distanceSq= 0.f;

	for (int axis= 0; axis < 3; ++axis)
	{
		const float value= p[axis];

		if (value < aabb_min[axis])
		{
			const float delta= aabb_min[axis] - value;
			distanceSq+= delta * delta;
		}
		else if (value > aabb_max[axis])
		{
			const float delta= value - aabb_max[axis];
			distanceSq+= delta * delta;
		}
	}

	return distanceSq;
}

bool glm_intersect_disk_with_ray(
	const glm::vec3& ray_start,     // Ray origin, in world space
	const glm::vec3& ray_direction, // Ray direction, in world space.
	const glm::vec3& disk_center, const glm::vec3& disk_normal, const float disk_radius,
	float& outIntDistance,   // Output: distance between ray_origin and the intersection with the OBB
	glm::vec3& outIntPoint,  // Output: intersection point on the surface of the OBB
	glm::vec3& outIntNormal) // Output: intersection normal on the surface of the OBB
{
	const glm::vec3 ray_unit_direction= glm::normalize(ray_direction);

	float intDistance= 0.f;
	if (glm::intersectRayPlane(ray_start, ray_unit_direction, disk_center, disk_normal, intDistance))
	{
		const glm::vec3 intPoint= ray_start + ray_direction * intDistance;
		const float intRadiusSqrd= glm::distance2(intPoint, disk_center);

		if (intRadiusSqrd <= disk_radius * disk_radius)
		{
			outIntDistance= intDistance;
			outIntNormal= disk_normal;
			outIntPoint= intPoint;
			return true;
		}
	}

	return false;
}

// Adapted from: https://github.com/opengl-tutorials/ogl/blob/master/misc05_picking/misc05_picking_custom.cpp
bool glm_intersect_obb_with_ray(
	const glm::vec3& ray_start,     // Ray origin, in world space
	const glm::vec3& ray_direction, // Ray direction, in world space.
	const glm::vec3& aabb_min,      // Minimum X,Y,Z coords of the mesh when not transformed at all.
	const glm::vec3&
		aabb_max, // Maximum X,Y,Z coords. Often aabb_min*-1 if your mesh is centered, but it's not always the case.
	const glm::mat4& xform,  // Transformation applied to the mesh (which will thus be also applied to its bounding box)
	float& outIntDistance,   // Output: distance between ray_origin and the intersection with the OBB
	glm::vec3& outIntPoint,  // Output: intersection point on the surface of the OBB
	glm::vec3& outIntNormal) // Output: intersection normal on the surface of the OBB
{
	const glm::vec3 ray_unit_direction= glm::normalize(ray_direction);

	// Intersection method from Real-Time Rendering and Essential Mathematics for Games
	float tMin= 0.0f;
	float tMax= k_real_max;
	glm::vec3 normal(0.f);

	glm::vec3 obb_center(xform[3]);
	glm::vec3 delta= obb_center - ray_start;

	// Test intersection with the 2 planes perpendicular to the OBB's X axis
	{
		glm::vec3 xaxis(xform[0]);
		float e= glm::dot(xaxis, delta);
		float f= glm::dot(ray_unit_direction, xaxis);

		if (fabs(f) > 0.001f)
		{ // Standard case

			float t1= (e + aabb_min.x) / f; // Intersection with the "left" plane
			float t2= (e + aabb_max.x) / f; // Intersection with the "right" plane
			// t1 and t2 now contain distances betwen ray origin and ray-plane intersections

			// We want t1 to represent the nearest intersection,
			// so if it's not the case, invert t1 and t2
			if (t1 > t2)
			{
				float w= t1;
				t1= t2;
				t2= w; // swap t1 and t2
			}

			// tMax is the nearest "far" intersection (amongst the X,Y and Z planes pairs)
			if (t2 < tMax)
				tMax= t2;
			// tMin is the farthest "near" intersection (amongst the X,Y and Z planes pairs)
			if (t1 > tMin)
				tMin= t1;

			// And here's the trick :
			// If "far" is closer than "near", then there is NO intersection.
			// See the images in the tutorials for the visual explanation.
			if (tMax < tMin)
				return false;

			// Collision normal is a vector on the x-axis point away from the
			normal= xaxis * sgn(e) * -1.f;
		}
		else
		{ // Rare case : the ray is almost parallel to the planes, so they don't have any "intersection"
			if (-e + aabb_min.x > 0.0f || -e + aabb_max.x < 0.0f)
				return false;
		}
	}

	// Test intersection with the 2 planes perpendicular to the OBB's Y axis
	// Exactly the same thing than above.
	{
		glm::vec3 yaxis(xform[1]);
		float e= glm::dot(yaxis, delta);
		float f= glm::dot(ray_unit_direction, yaxis);

		if (fabs(f) > 0.001f)
		{

			float t1= (e + aabb_min.y) / f;
			float t2= (e + aabb_max.y) / f;

			if (t1 > t2)
			{
				float w= t1;
				t1= t2;
				t2= w;
			}

			if (t2 < tMax)
				tMax= t2;

			if (t1 > tMin)
			{
				tMin= t1;
				normal= yaxis * sgn(e) * -1.f;
			}

			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + aabb_min.y > 0.0f || -e + aabb_max.y < 0.0f)
				return false;
		}
	}

	// Test intersection with the 2 planes perpendicular to the OBB's Z axis
	// Exactly the same thing than above.
	{
		glm::vec3 zaxis(xform[2]);
		float e= glm::dot(zaxis, delta);
		float f= glm::dot(ray_unit_direction, zaxis);

		if (fabs(f) > 0.001f)
		{

			float t1= (e + aabb_min.z) / f;
			float t2= (e + aabb_max.z) / f;

			if (t1 > t2)
			{
				float w= t1;
				t1= t2;
				t2= w;
			}

			if (t2 < tMax)
				tMax= t2;

			if (t1 > tMin)
			{
				tMin= t1;
				normal= zaxis * sgn(e) * -1.f;
			}

			if (tMin > tMax)
				return false;
		}
		else
		{
			if (-e + aabb_min.z > 0.0f || -e + aabb_max.z < 0.0f)
				return false;
		}
	}

	outIntDistance= tMin;
	outIntPoint= ray_start + ray_direction * outIntDistance;
	outIntNormal= (outIntDistance > k_normal_epsilon) ? normal : -ray_unit_direction;

	return true;
}

bool glm_intersect_aabb_with_ray(const glm::vec3& ray_start, const glm::vec3& ray_direction, const glm::vec3& aabb_min,
								 const glm::vec3& aabb_max, float& outIntDistance)
{
	const glm::vec3 tMin= (aabb_min - ray_start) / ray_direction;
	const glm::vec3 tMax= (aabb_max - ray_start) / ray_direction;
	const glm::vec3 t1= glm::min(tMin, tMax);
	const glm::vec3 t2= glm::max(tMin, tMax);
	const float tNear= glm::max(glm::max(t1.x, t1.y), t1.z);
	const float tFar= glm::min(glm::min(t2.x, t2.y), t2.z);

	outIntDistance= glm::max(tNear, 0.f);

	return tFar >= tNear && (tNear >= 0.f || tFar >= 0.f);
}

// https://stackoverflow.com/questions/33532860/merge-two-spheres-to-get-a-new-one
void glm_sphere_union(const glm::vec3& c1, const float r1, const glm::vec3& c2, const float r2, glm::vec3& outC,
					  float& outR)
{
	const glm::vec3 c1_to_c2= c2 - c1;
	const float dist= glm::length(c1_to_c2);

	// Is sphere 1 completely inside sphere 2?
	if (dist + r1 <= r2)
	{
		outC= c2;
		outR= r2;
	}
	// Is sphere 2 completely inside sphere 1?
	else if (dist + r2 <= r1)
	{
		outC= c1;
		outR= r1;
	}
	// Otherwise compute the new bounding sphere that overlaps both
	else
	{
		const float R= (r1 + r2 + dist) / 2.f;

		outC= c1 + (c1_to_c2 * (R - r1) / dist);
		outR= R;
	}
}