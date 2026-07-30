//-- includes -----
#include "MathUtility.h"

//-- int methods -----
int int_min(int a, int b) { return (a < b) ? a : b; }

int int_max(int a, int b) { return (a > b) ? a : b; }

int int_clamp(int x, int lo, int hi) { return int_min(int_max(x, lo), hi); }

//-- float methods -----
float safe_divide_with_default(float numerator, float denomenator, float default_result)
{
	return is_nearly_zero(denomenator) ? default_result : (numerator / denomenator);
}

float safe_sqrt_with_default(float square, float default_result)
{
	return is_nearly_zero(square) ? default_result : sqrtf(square);
}

float remap_float_to_float(float inA, float inB, float outA, float outB, float inValue)
{
	if (fabsf(inB - inA) > k_real_epsilon)
	{
		if (inB > inA)
		{
			float clampedValue= fmaxf(fminf(inValue, inB), inA);
			float u= (clampedValue - inA) / (inB - inA);
			float rempappedValue= ((1.f - u) * outA + u * outB);

			return rempappedValue;
		}
		else
		{
			float clampedValue= fmaxf(fminf(inValue, inA), inB);
			float u= (clampedValue - inB) / (inA - inB);
			float rempappedValue= (u * outA + (1.f - u) * outB);

			return rempappedValue;
		}
	}
	else
	{
		// If the input range is zero, return the average of the output range
		return (outA + outB) * 0.5f;
	}
}

int remap_float_to_int(float inA, float inB, int outA, int outB, float inValue)
{
	return (int)remap_float_to_float(inA, inB, (float)outA, (float)outB, inValue);
}

float remap_int_to_float(int intA, int intB, float outA, float outB, int inValue)
{
	return remap_float_to_float((float)intA, (float)intB, outA, outB, (float)inValue);
}

int remap_int_to_int(int inA, int inB, int outA, int outB, int inValue)
{
	return (int)remap_float_to_float((float)inA, (float)inB, (float)outA, (float)outB, (float)inValue);
}

float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

double clampd(double x, double lo, double hi) { return fmin(fmax(x, lo), hi); }

float clampf01(float x) { return clampf(x, 0.f, 1.f); }

float lerpf(float a, float b, float u) { return a * (1.f - u) + b * u; }

float lerp_clampf(float a, float b, float u) { return clampf(lerpf(a, b, u), a, b); }

float degrees_to_radians(float x) { return ((x * k_real_pi) / 180.f); }

float radians_to_degrees(float x) { return ((x * 180.f) / k_real_pi); }

float wrap_radians(float angle) { return fmodf(angle + k_real_two_pi, k_real_two_pi); }

float wrap_degrees(float angle) { return fmodf(angle + 360.f, 360.f); }

float wrap_range(float value, float range_min, float range_max)
{
	assert(range_max > range_min);
	const float range= range_max - range_min;

	return range_min + fmodf((value - range_min) + range, range);
}

double wrap_ranged(double value, double range_min, double range_max)
{
	assert(range_max > range_min);
	const double range= range_max - range_min;

	return range_min + fmod((value - range_min) + range, range);
}

float wrap_lerpf(float a, float b, float u, float range_min, float range_max)
{
	assert(range_max > range_min);
	const float range= range_max - range_min;
	float wrapped_a= a;
	float wrapped_b= b;

	if (fabsf(a - b) >= (range / 2.f))
	{
		if (a > b)
			wrapped_a= wrap_range(a, range_min, range_max) - range;
		else
			wrapped_b= wrap_range(b, range_min, range_max) - range;
	}

	return wrap_range(lerpf(wrapped_a, wrapped_b, u), range_min, range_max);
}