public void modifyDirective(CacheDirectiveInfo info,
      FSPermissionChecker pc, EnumSet<CacheFlag> flags) throws IOException {
    assert namesystem.hasWriteLock();
    String idString =
        (info.getId() == null) ?
            "(null)" : info.getId().toString();
    try {
      // Check for invalid IDs.
      Long id = info.getId();
      if (id == null) {
        throw new InvalidRequestException("Must supply an ID.");
      }
      CacheDirective prevEntry = getById(id);
      checkWritePermission(pc, prevEntry.getPool());

      // Fill in defaults
      CacheDirectiveInfo infoWithDefaults =
          createFromInfoAndDefaults(info, prevEntry);
      CacheDirectiveInfo.Builder builder =
          new CacheDirectiveInfo.Builder(infoWithDefaults);

      // Do validation
      validatePath(infoWithDefaults);
      validateReplication(infoWithDefaults, (short)-1);
      // Need to test the pool being set here to avoid rejecting a modify for a
      // directive that's already been forced into a pool
      CachePool srcPool = prevEntry.getPool();
      CachePool destPool = getCachePool(validatePoolName(infoWithDefaults));
      if (!srcPool.getPoolName().equals(destPool.getPoolName())) {
        checkWritePermission(pc, destPool);
        if (!flags.contains(CacheFlag.FORCE)) {
          checkLimit(destPool, infoWithDefaults.getPath().toUri().getPath(),
              infoWithDefaults.getReplication());
        }
      }
      // Verify the expiration against the destination pool
      validateExpiryTime(infoWithDefaults, destPool.getMaxRelativeExpiryMs());

      // Indicate changes to the CRM
      setNeedsRescan();

      // Validation passed
      removeInternal(prevEntry);
      addInternal(new CacheDirective(builder.build()), destPool);
    } catch (IOException e) {
      LOG.warn("modifyDirective of " + idString + " failed: ", e);
      throw e;
    }
    LOG.info("modifyDirective of {} successfully applied {}.", idString, info);
  }