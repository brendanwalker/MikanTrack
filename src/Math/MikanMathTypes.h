#pragma once

// Plain-struct retype of MikanXR's MikanClientAPI math types
// (MikanXR: src/Libraries/MikanClientAPI/Public/MikanMathTypes.h),
// with the Refureku reflection / serialization machinery removed.
// Field layout and semantics are identical to the originals.

//-- constants -----
/// Conversion factor to go from meters to centimeters
#define MIKAN_METERS_TO_CENTIMETERS 100.f

/// Conversion factor to go from centimeters to meters
#define MIKAN_CENTIMETERS_TO_METERS 0.01f

/// A 2D vector with int components.
struct MikanVector2i
{
	int x= 0;
	int y= 0;
};

/// A 2D vector with float components.
struct MikanVector2f
{
	float x= 0;
	float y= 0;
};

/// A 2D vector with double components.
struct MikanVector2d
{
	double x= 0;
	double y= 0;
};

/// A 3D vector with float components.
struct MikanVector3f
{
	float x= 0;
	float y= 0;
	float z= 0;
};

/// A 3D vector with double components.
struct MikanVector3d
{
	double x= 0;
	double y= 0;
	double z= 0;
};

/// A 3-tuple of Euler angles with float components.
struct MikanRotator3f
{
	float x_angle= 0;
	float y_angle= 0;
	float z_angle= 0;
};

/// A 4D vector with float components.
struct MikanVector4f
{
	float x= 0;
	float y= 0;
	float z= 0;
	float w= 0;
};

/// A 4D vector with double components.
struct MikanVector4d
{
	double x= 0;
	double y= 0;
	double z= 0;
	double w= 0;
};

/** A 4x4 matrix with float components
	storage is column major order:

	| x0 y0 z0 w0 |
	| x1 y1 z1 w1 |
	| x2 y2 z2 w2 |
	| x3 y3 z3 w3 |
 */
struct MikanMatrix4f
{
	float x0= 1;
	float x1= 0;
	float x2= 0;
	float x3= 0;
	float y0= 0;
	float y1= 1;
	float y2= 0;
	float y3= 0;
	float z0= 0;
	float z1= 0;
	float z2= 1;
	float z3= 0;
	float w0= 0;
	float w1= 0;
	float w2= 0;
	float w3= 1;
};

/** A 3x3 matrix with double components
	storage is column major order:

	| x0 y0 z0 |
	| x1 y1 z1 |
	| x2 y2 z2 |
 */
struct MikanMatrix3d
{
	double x0= 1;
	double x1= 0;
	double x2= 0;
	double y0= 0;
	double y1= 1;
	double y2= 0;
	double z0= 0;
	double z1= 0;
	double z2= 1;
};

/** A 4x3 matrix with double components
	storage is column major order:

	| x0 y0 z0 w0|
	| x1 y1 z1 w1|
	| x2 y2 z2 w2|
 */
struct MikanMatrix4x3d
{
	double x0= 1;
	double x1= 0;
	double x2= 0;
	double x3= 0;
	double y0= 0;
	double y1= 1;
	double y2= 0;
	double y3= 0;
	double z0= 0;
	double z1= 0;
	double z2= 1;
	double z3= 0;
};

/** A 4x4 matrix with double components
	storage is column major order:

	| x0 y0 z0 w0 |
	| x1 y1 z1 w1 |
	| x2 y2 z2 w2 |
	| x3 y3 z3 w3 |
 */
struct MikanMatrix4d
{
	double x0= 1;
	double x1= 0;
	double x2= 0;
	double x3= 0;
	double y0= 0;
	double y1= 1;
	double y2= 0;
	double y3= 0;
	double z0= 0;
	double z1= 0;
	double z2= 1;
	double z3= 0;
	double w0= 0;
	double w1= 0;
	double w2= 0;
	double w3= 1;
};

/// A single-precision quaternion rotation.
struct MikanQuatf
{
	float w= 1;
	float x= 0;
	float y= 0;
	float z= 0;
};

/// A double-precision quaternion rotation.
struct MikanQuatd
{
	double w= 1;
	double x= 0;
	double y= 0;
	double z= 0;
};

/// A single-precision Scale-Rotation-Translation transform.
struct MikanTransform
{
	MikanVector3f scale;
	MikanQuatf rotation;
	MikanVector3f position;
};
