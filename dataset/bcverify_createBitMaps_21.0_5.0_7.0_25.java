public static void createBitMaps() {
		int i;
		long j, k, bitPosition;
		char c;
		// Create the valid character bit maps for the first BIT_MAP_END characters
		j = 0;
		k = 0;
		bitPosition = 1;
		for (i = 0; i < BIT_MAP_END; i++) {
			c = (char) i;
			if (Character.isJavaIdentifierStart(c)) {
				j |= bitPosition;
			}
			if (Character.isJavaIdentifierPart(c)) {
				k |= bitPosition;
			}
			bitPosition <<= 1;
			if (bitPosition == 4294967296L) {
				bitMapStartChars[i >> 5] = j;
				bitMapPartChars[i >> 5] = k;
				j = 0;
				k = 0;
				bitPosition = 1;
			}
		}
	}