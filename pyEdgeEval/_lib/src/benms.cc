#include <math.h>
#include <assert.h>
#include "Matrix.hh"
#include "benms.hh"

/*******************************************************************************
* Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/

inline double interp(const Matrix& I, int h, int w, double x, double y)
{
  // return I[x,y] via bilinear interpolation
  x = x < 0 ? 0 : (x > w - 1.001 ? w - 1.001 : x);
  y = y < 0 ? 0 : (y > h - 1.001 ? h - 1.001 : y);
  int x0 = int(x), y0 = int(y);
  int x1 = x0 + 1, y1 = y0 + 1;
  double dx0 = x - x0, dy0 = y - y0;
  double dx1 = 1 - dx0, dy1 = 1 - dy0;
  double out = I(y0, x0) * dx1 * dy1 +
               I(y0, x1) * dx0 * dy1 +
               I(y1, x0) * dx1 * dy0 +
               I(y1, x1) * dx0 * dy0;
  return out;
}

void benms(Matrix& out, const Matrix& edge, const Matrix& ori, int r, int s, double m)
{
  // r: radius for nms supr
  // s: radius for supr boundaries
  // m: multiplier for conservative supr

  assert (edge.ncols() == ori.ncols());
  assert (edge.nrows() == ori.nrows());

  int w = edge.ncols();
  int h = edge.nrows();

  out = Matrix(h, w);

  // suppress edges where edge is stronger in orthogonal direction
  for (int x = 0; x < w; ++x) {
    for (int y = 0; y < h; ++y) {
      double e = out(y, x) = edge(y, x);
      if (e == 0) {
        continue;
      }
      e *= m;
      double cos_o = cos(ori(y, x));
      double sin_o = sin(ori(y, x));
      for (int d = -r; d <= r; ++d) {
        if (d != 0) {
          double e0 = interp(edge, h, w, x + d * cos_o, y + d * sin_o);
          if (e < e0) {
            out(y, x) = 0;
            break;
          }
        }
      }
    }
  }

  // suppress noisy edge estimates near boundaries
  s = s > w / 2 ? w / 2 : s;
  s = s > h / 2 ? h / 2 : s;
  for (int x = 0; x < s; ++x) {
    for (int y = 0; y < h; ++y) {
      out(y, x) *= double(x) / s;
      out(y, w - 1 - x) *= double(x) / s;
    }
  }
  for (int x = 0; x < w; ++x) {
    for (int y = 0; y < s; ++y) {
      out(y, x) *= double(y) / s;
      out(h - 1 - y, x) *= double(y) / s;
    }
  }
}
