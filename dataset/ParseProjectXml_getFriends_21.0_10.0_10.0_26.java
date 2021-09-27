private String[] getFriends(Document d) throws BuildException {
        Element cfg = getConfig(d);
        Element pp = findNBMElement(cfg, "friend-packages");
        if (pp == null) {
            return null;
        }
        List<String> friends = new ArrayList<>();
        boolean other = false;
        for (Element p : XMLUtil.findSubElements(pp)) {
            if ("friend".equals(p.getNodeName())) {
                String t = XMLUtil.findText(p);
                if (t == null) {
                    throw new BuildException("No text in <friend>", getLocation());
                }
                friends.add(t);
            } else {
                other = true;
            }
        }
        if (friends.isEmpty()) {
            throw new BuildException("Must have at least one <friend> in <friend-packages>", getLocation());
        }
        if (!other) {
            throw new BuildException("Must have at least one <package> in <friend-packages>", getLocation());
        }
        return friends.toArray(new String[friends.size()]);
    }