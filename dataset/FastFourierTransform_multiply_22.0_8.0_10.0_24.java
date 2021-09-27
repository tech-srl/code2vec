public static long[] multiply(long[] x, long[] y) {

    // If the coefficients are negative place them in the range of [0, p)
    for (int i = 0; i < x.length; i++) if (x[i] < 0) x[i] += p;
    for (int i = 0; i < y.length; i++) if (y[i] < 0) y[i] += p;

    int zLength = x.length + y.length - 1;
    int logN = 32 - Integer.numberOfLeadingZeros(zLength - 1);
    long[] xx = transform(x, logN, false);
    long[] yy = transform(y, logN, false);
    long[] zz = new long[1 << logN];
    for (int i = 0; i < zz.length; i++) zz[i] = mult(xx[i], yy[i]);
    long[] nZ = transform(zz, logN, true);
    long[] z = new long[zLength];
    long nInverse = p - ((p - 1) >>> logN);
    for (int i = 0; i < z.length; i++) {

      z[i] = mult(nInverse, nZ[i]);

      // Allow for negative coefficients. If you know the answer cannot be
      // greater than 2^31-1 subtract p to obtain the negative coefficient.
      if (z[i] >= Integer.MAX_VALUE) z[i] -= p;
    }
    return z;
  }