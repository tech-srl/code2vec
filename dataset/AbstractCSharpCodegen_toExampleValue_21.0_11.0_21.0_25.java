@Override
    public String toExampleValue(Schema p) {
        if (ModelUtils.isStringSchema(p)) {
            if (p.getExample() != null) {
                return "\"" + p.getExample().toString() + "\"";
            }
        } else if (ModelUtils.isBooleanSchema(p)) {
            if (p.getExample() != null) {
                return p.getExample().toString();
            }
        } else if (ModelUtils.isDateSchema(p)) {
            // TODO
        } else if (ModelUtils.isDateTimeSchema(p)) {
            // TODO
        } else if (ModelUtils.isNumberSchema(p)) {
            if (p.getExample() != null) {
                return p.getExample().toString();
            }
        } else if (ModelUtils.isIntegerSchema(p)) {
            if (p.getExample() != null) {
                return p.getExample().toString();
            }
        }

        return null;
    }