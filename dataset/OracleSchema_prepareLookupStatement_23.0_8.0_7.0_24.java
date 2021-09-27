@NotNull
        @Override
        public JDBCStatement prepareLookupStatement(@NotNull JDBCSession session, @NotNull OracleSchema owner, @Nullable OracleTableBase object, @Nullable String objectName) throws SQLException {
            String tableOper = "=";

            boolean hasAllAllTables = owner.getDataSource().isViewAvailable(session.getProgressMonitor(), null, "ALL_ALL_TABLES");
            String tablesSource = hasAllAllTables ? "ALL_TABLES" : "TABLES";
            String tableTypeColumns = hasAllAllTables ? "t.TABLE_TYPE_OWNER,t.TABLE_TYPE" : "NULL as TABLE_TYPE_OWNER, NULL as TABLE_TYPE";

            final JDBCPreparedStatement dbStat = session.prepareStatement(
                "\tSELECT " + OracleUtils.getSysCatalogHint(owner.getDataSource()) + " t.OWNER,t.TABLE_NAME as TABLE_NAME,'TABLE' as OBJECT_TYPE,'VALID' as STATUS," + tableTypeColumns + ",t.TABLESPACE_NAME,t.PARTITIONED,t.IOT_TYPE,t.IOT_NAME,t.TEMPORARY,t.SECONDARY,t.NESTED,t.NUM_ROWS \n" +
                    "\tFROM " + OracleUtils.getAdminAllViewPrefix(session.getProgressMonitor(), owner.getDataSource(), tablesSource) + " t\n" +
                    "\tWHERE t.OWNER=? AND NESTED='NO'" + (object == null && objectName == null ? "": " AND t.TABLE_NAME"+ tableOper + "?") + "\n" +
                "UNION ALL\n" +
                    "\tSELECT " + OracleUtils.getSysCatalogHint(owner.getDataSource()) + " o.OWNER,o.OBJECT_NAME as TABLE_NAME,'VIEW' as OBJECT_TYPE,o.STATUS,NULL,NULL,NULL,'NO',NULL,NULL,o.TEMPORARY,o.SECONDARY,'NO',0 \n" +
                    "\tFROM " + OracleUtils.getAdminAllViewPrefix(session.getProgressMonitor(), owner.getDataSource(), "OBJECTS") + " o \n" +
                    "\tWHERE o.OWNER=? AND o.OBJECT_TYPE='VIEW'" + (object == null && objectName == null  ? "": " AND o.OBJECT_NAME" + tableOper + "?") + "\n"
                );
            int index = 1;
            dbStat.setString(index++, owner.getName());
            if (object != null || objectName != null) dbStat.setString(index++, object != null ? object.getName() : objectName);
            dbStat.setString(index++, owner.getName());
            if (object != null || objectName != null) dbStat.setString(index, object != null ? object.getName() : objectName);
            return dbStat;
        }