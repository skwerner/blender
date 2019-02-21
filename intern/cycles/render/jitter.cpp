/*
 * Copyright 2019 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "render/jitter.h"
#include <math.h>
#include <vector>

CCL_NAMESPACE_BEGIN

static float rnd()
{
	return drand48();
}

float2 generate_sample_point(float i, float j, float xhalf, float yhalf, float n)
{
	float2 pt;
	pt.x = (i + 0.5f * (xhalf + rnd())) / n;
	pt.y = (j + 0.5f * (yhalf + rnd())) / n;
	return pt;
}

void extend_sequence(float2 points[], int N)
{
	int n = sqrtf(N);
	for(int s = 0; s < N; ++s) {
		float2 oldpt = points[s];
		float i = floorf(n * oldpt.x);
		float j = floorf(n * oldpt.y);
		float xhalf = floorf(2.0f * (n * oldpt.x - i));
		float yhalf = floorf(2.0f * (n * oldpt.y - j));
		xhalf = 1.0f - xhalf;
		yhalf = 1.0f - yhalf;
		points[N + s] = generate_sample_point(i, j, xhalf, yhalf, n);
		if(rnd() > 0.5f) {
			xhalf = 1.0f - xhalf;
		}
		else {
			yhalf = 1.0f - yhalf;
		}
		points[2 * N + s] = generate_sample_point(i, j, xhalf, yhalf, n);
		xhalf = 1.0f - xhalf;
		yhalf = 1.0f - yhalf;
		points[3 * N + s] = generate_sample_point(i, j, xhalf, yhalf, n);
	}
}

float2 *generate_pj(int M)
{
	float2 *points = new float2[M];
	return points;
}

void progressive_jitter_generate_2D(float2 points[], int size)
{
	points[0].x = rnd();
	points[0].y = rnd();
	int N = 1;
	while(N < size) {
		extend_sequence(points, N);
		N = 4 * N;
	}
}

class PMJ_Generator
{
public:
	static void generate_2D(float2 points[], int size)
	{
		points[0].x = rnd();
		points[0].y = rnd();
		int N = 1;
		PMJ_Generator g;
		while(N < size) {
			g.extend_sequence_even(points, N);
			g.extend_sequence_odd(points, 2 * N);
			N = 4 * N;
		}
	}
protected:
	PMJ_Generator() : num_samples(1) {}
	virtual void mark_occupied_strata(float2 points[], int N)
	{
		int NN = 2 * N;
		for(int s = 0; s < NN; ++s) {
			occupied1Dx[s] = occupied1Dy[s] = false;
		}
		for(int s = 0; s < N; ++s) {
			int xstratum = std::min(int(NN * points[s].x), NN-1);
			int ystratum = std::min(int(NN * points[s].y), NN-1);
			occupied1Dx[xstratum] = true;
			occupied1Dy[ystratum] = true;
		}
	}

	virtual void generate_sample_point(float2 points[], float i, float j, float xhalf, float yhalf, int n, int N)
	{
		int NN = 2 * N;
		float2 pt;
		int xstratum, ystratum;
		do {
			pt.x = (i + 0.5f * (xhalf + rnd())) / n;
			xstratum = std::min(int(NN * pt.x), NN-1);
		} while(occupied1Dx[xstratum]);
		do {
			pt.y = (j + 0.5f * (yhalf + rnd())) / n;
			ystratum = std::min(int(NN * pt.y), NN-1);
		} while(occupied1Dy[ystratum]);
		occupied1Dx[xstratum] = true;
		occupied1Dy[ystratum] = true;
		points[num_samples] = pt;
		++num_samples;
	}

	void extend_sequence_even(float2 points[], int N)
	{
		int n = sqrtf(N);
		occupied1Dx.resize(2*N);
		occupied1Dy.resize(2*N);
		mark_occupied_strata(points, N);
		for(int s = 0; s < N; ++s) {
			float2 oldpt = points[s];
			float i = floorf(n * oldpt.x);
			float j = floorf(n * oldpt.y);
			float xhalf = floorf(2.0f * (n * oldpt.x - i));
			float yhalf = floorf(2.0f * (n * oldpt.y - j));
			xhalf = 1.0f - xhalf;
			yhalf = 1.0f - yhalf;
			generate_sample_point(points, i, j, xhalf, yhalf, n, N);
		}
	}

	void extend_sequence_odd(float2 points[], int N)
	{
		int n = sqrtf(N/2);
		occupied1Dx.resize(2*N);
		occupied1Dy.resize(2*N);
		mark_occupied_strata(points,N);
		std::vector<float> xhalves(N/2);
		std::vector<float> yhalves(N/2);
		for(int s = 0; s < N/2; ++s) {
			float2 oldpt = points[s];
			float i = floorf(n * oldpt.x);
			float j = floorf(n * oldpt.y);
			float xhalf = floorf(2.0f * (n * oldpt.x - i));
			float yhalf = floorf(2.0f * (n * oldpt.y - j));
			if(rnd() > 0.5f) {
				xhalf = 1.0f - xhalf;
			}
			else {
				yhalf = 1.0f - yhalf;
			}
			xhalves[s] = xhalf;
			yhalves[s] = yhalf;
			generate_sample_point(points, i, j, xhalf, yhalf, n, N);
		}
		for(int s = 0; s < N/2; ++s) {
			float2 oldpt = points[s];
			float i = floorf(n * oldpt.x);
			float j = floorf(n * oldpt.y);
			float xhalf = 1.0f - xhalves[s];
			float yhalf = 1.0f - yhalves[s];
			generate_sample_point(points, i, j, xhalf, yhalf, n, N);
		}
	}

	std::vector<bool> occupied1Dx, occupied1Dy;
	int num_samples;
};

void progressive_multi_jitter_generate_2D(float2 points[], int size)
{
	PMJ_Generator::generate_2D(points, size);
}

class PMJ02_Generator : public PMJ_Generator
{
public:
	void mark_occupied_strata(float2 points[], int N) override
	{
		int NN = 2 * N;
		int num_shapes = log2f(NN);
		occupiedStrata.resize(num_shapes, std::vector<bool>(NN));
		/* is it all false by default? */
		for(int shape = 0; shape < num_shapes; ++shape) {
			for(int n = 0; n < NN; ++n) {
				occupiedStrata[shape][n] = false;
			}
		}
		for(int s = 0; s < N; ++s) {
			mark_occupied_strata1(points[s], NN);
		}
	}

	void mark_occupied_strata1(float2 pt, int NN)
	{
		int shape = 0;
		int xdivs = NN;
		int ydivs = 1;
		do {
			int xstratum = xdivs * pt.x;
			int ystratum = ydivs * pt.y;
			occupiedStrata[shape][ystratum & xdivs + xstratum] = true;
			shape = shape + 1;
			xdivs = xdivs / 2;
			ydivs = ydivs * 2;
		} while(xdivs > 0);
	}

	void generate_sample_point(float2 points[], float i, float j, float xhalf, float yhalf, int n, int N) override
	{
		int NN = 2 * N;
		float2 pt;
		do {
			pt.x = (i + 0.5f * (xhalf + rnd())) / n;
			pt.y = (j + 0.5f * (yhalf + rnd()))/ n;
		} while(is_occupied(pt, NN));
		mark_occupied_strata1(pt, NN);
		points[num_samples] = pt;
		++num_samples;
	}

	bool is_occupied(float2 pt, int NN)
	{
		int shape = 0;
		int xdivs = NN;
		int ydivs = 1;
		do {
			int xstratum = xdivs * pt.x;
			int ystratum = ydivs * pt.y;
			if(occupiedStrata[shape][ystratum * xdivs + xstratum]) {
				return true;
			}
			shape = shape + 1;
			xdivs = xdivs / 2;
			ydivs = ydivs * 2;
		} while (xdivs > 0);
		return false;
	}

private:
	std::vector<std::vector<bool>> occupiedStrata;
};

void progressive_multi_jitter_02_generate_2D(float2 points[], int size)
{
	PMJ02_Generator::generate_2D(points, size);
}

CCL_NAMESPACE_END
