private void basicItem(FireBirdPlanNode parent) throws FireBirdPlanException {
			String aliases = collectIdentifiers();
			switch (tokenMatch.token) {
				case NATURAL:
					addPlanNode(parent, aliases + " NATURAL");
					tokenMatch.jump();
					break;
				case INDEX:
					String indexes = collectIndexes();
					addPlanNode(parent, aliases + " INDEX (" + indexes + ")");
					break;
				case ORDER:
					tokenMatch.jump();
					tokenMatch.checkToken(FireBirdPlanToken.IDENTIFICATOR);
					String orderIndex = tokenMatch.getValue();
					tokenMatch.jump();
					String text = aliases + " ORDER " + orderIndex + indexInfo(orderIndex);
					if (tokenMatch.getToken() == FireBirdPlanToken.INDEX) {
						String orderIndexes = collectIndexes();
						text = text + " INDEX(" + orderIndexes + ")";
					}
					addPlanNode(parent, text);
					break;
			default:
				tokenMatch.raisePlanTokenException();
			}
			
		}