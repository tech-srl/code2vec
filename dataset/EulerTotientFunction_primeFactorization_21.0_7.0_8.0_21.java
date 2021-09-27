private static ArrayList<Long> primeFactorization(long n) {
    ArrayList<Long> factors = new ArrayList<Long>();
    if (n <= 0) throw new IllegalArgumentException();
    else if (n == 1) return factors;
    PriorityQueue<Long> divisorQueue = new PriorityQueue<Long>();
    divisorQueue.add(n);
    while (!divisorQueue.isEmpty()) {
      long divisor = divisorQueue.remove();
      if (isPrime(divisor)) {
        factors.add(divisor);
        continue;
      }
      long next_divisor = pollardRho(divisor);
      if (next_divisor == divisor) {
        divisorQueue.add(divisor);
      } else {
        divisorQueue.add(next_divisor);
        divisorQueue.add(divisor / next_divisor);
      }
    }
    return factors;
  }