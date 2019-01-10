package JavaExtractor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.StringJoiner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseException;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import JavaExtractor.FeaturesEntities.Property;
import JavaExtractor.Visitors.FunctionVisitor;

@SuppressWarnings("StringEquality")
public class FeatureExtractor {
	private CommandLineValues m_CommandLineValues;
	private static Set<String> s_ParentTypeToAddChildId = Stream
			.of("AssignExpr", "ArrayAccessExpr", "FieldAccessExpr", "MethodCallExpr")
			.collect(Collectors.toCollection(HashSet::new));

	final static String lparen = "(";
	final static String rparen = ")";
	final static String upSymbol = "^";
	final static String downSymbol = "_";

	public FeatureExtractor(CommandLineValues commandLineValues) {
		this.m_CommandLineValues = commandLineValues;
	}

	public ArrayList<ProgramFeatures> extractFeatures(String code) throws ParseException, IOException {
		CompilationUnit compilationUnit = parseFileWithRetries(code);
		FunctionVisitor functionVisitor = new FunctionVisitor();

		functionVisitor.visit(compilationUnit, null);

		ArrayList<MethodContent> methods = functionVisitor.getMethodContents();
		ArrayList<ProgramFeatures> programs = generatePathFeatures(methods);

		return programs;
	}

	private CompilationUnit parseFileWithRetries(String code) throws IOException {
		final String classPrefix = "public class Test {";
		final String classSuffix = "}";
		final String methodPrefix = "SomeUnknownReturnType f() {";
		final String methodSuffix = "return noSuchReturnValue; }";

		String originalContent = code;
		String content = originalContent;
		CompilationUnit parsed = null;
		try {
			parsed = JavaParser.parse(content);
		} catch (ParseProblemException e1) {
			// Wrap with a class and method
			try {
				content = classPrefix + methodPrefix + originalContent + methodSuffix + classSuffix;
				parsed = JavaParser.parse(content);
			} catch (ParseProblemException e2) {
				// Wrap with a class only
				content = classPrefix + originalContent + classSuffix;
				parsed = JavaParser.parse(content);
			}
		}

		return parsed;
	}

	public ArrayList<ProgramFeatures> generatePathFeatures(ArrayList<MethodContent> methods) {
		ArrayList<ProgramFeatures> methodsFeatures = new ArrayList<>();
		for (MethodContent content : methods) {
			if (content.getLength() < m_CommandLineValues.MinCodeLength
					|| content.getLength() > m_CommandLineValues.MaxCodeLength)
				continue;
			ProgramFeatures singleMethodFeatures = generatePathFeaturesForFunction(content);
			if (!singleMethodFeatures.isEmpty()) {
				methodsFeatures.add(singleMethodFeatures);
			}
		}
		return methodsFeatures;
	}

	private ProgramFeatures generatePathFeaturesForFunction(MethodContent methodContent) {
		ArrayList<Node> functionLeaves = methodContent.getLeaves();
		ProgramFeatures programFeatures = new ProgramFeatures(methodContent.getName());

		for (int i = 0; i < functionLeaves.size(); i++) {
			for (int j = i + 1; j < functionLeaves.size(); j++) {
				String separator = Common.EmptyString;

				String path = generatePath(functionLeaves.get(i), functionLeaves.get(j), separator);
				if (path != Common.EmptyString) {
					Property source = functionLeaves.get(i).getUserData(Common.PropertyKey);
					Property target = functionLeaves.get(j).getUserData(Common.PropertyKey);
					programFeatures.addFeature(source, path, target);
				}
			}
		}
		return programFeatures;
	}

	private static ArrayList<Node> getTreeStack(Node node) {
		ArrayList<Node> upStack = new ArrayList<>();
		Node current = node;
		while (current != null) {
			upStack.add(current);
			current = current.getParentNode();
		}
		return upStack;
	}

	private String generatePath(Node source, Node target, String separator) {
		String down = downSymbol;
		String up = upSymbol;
		String startSymbol = lparen;
		String endSymbol = rparen;

		StringJoiner stringBuilder = new StringJoiner(separator);
		ArrayList<Node> sourceStack = getTreeStack(source);
		ArrayList<Node> targetStack = getTreeStack(target);

		int commonPrefix = 0;
		int currentSourceAncestorIndex = sourceStack.size() - 1;
		int currentTargetAncestorIndex = targetStack.size() - 1;
		while (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0
				&& sourceStack.get(currentSourceAncestorIndex) == targetStack.get(currentTargetAncestorIndex)) {
			commonPrefix++;
			currentSourceAncestorIndex--;
			currentTargetAncestorIndex--;
		}

		int pathLength = sourceStack.size() + targetStack.size() - 2 * commonPrefix;
		if (pathLength > m_CommandLineValues.MaxPathLength) {
			return Common.EmptyString;
		}

		if (currentSourceAncestorIndex >= 0 && currentTargetAncestorIndex >= 0) {
			int pathWidth = targetStack.get(currentTargetAncestorIndex).getUserData(Common.ChildId)
					- sourceStack.get(currentSourceAncestorIndex).getUserData(Common.ChildId);
			if (pathWidth > m_CommandLineValues.MaxPathWidth) {
				return Common.EmptyString;
			}
		}

		for (int i = 0; i < sourceStack.size() - commonPrefix; i++) {
			Node currentNode = sourceStack.get(i);
			String childId = Common.EmptyString;
			String parentRawType = currentNode.getParentNode().getUserData(Common.PropertyKey).getRawType();
			if (i == 0 || s_ParentTypeToAddChildId.contains(parentRawType)) {
				childId = saturateChildId(currentNode.getUserData(Common.ChildId))
						.toString();
			}
			stringBuilder.add(String.format("%s%s%s%s%s", startSymbol,
					currentNode.getUserData(Common.PropertyKey).getType(), childId, endSymbol, up));
		}

		Node commonNode = sourceStack.get(sourceStack.size() - commonPrefix);
		String commonNodeChildId = Common.EmptyString;
		Property parentNodeProperty = commonNode.getParentNode().getUserData(Common.PropertyKey);
		String commonNodeParentRawType = Common.EmptyString;
		if (parentNodeProperty != null) {
			commonNodeParentRawType = parentNodeProperty.getRawType();
		}
		if (s_ParentTypeToAddChildId.contains(commonNodeParentRawType)) {
			commonNodeChildId = saturateChildId(commonNode.getUserData(Common.ChildId))
					.toString();
		}
		stringBuilder.add(String.format("%s%s%s%s", startSymbol,
				commonNode.getUserData(Common.PropertyKey).getType(), commonNodeChildId, endSymbol));

		for (int i = targetStack.size() - commonPrefix - 1; i >= 0; i--) {
			Node currentNode = targetStack.get(i);
			String childId = Common.EmptyString;
			if (i == 0 || s_ParentTypeToAddChildId.contains(currentNode.getUserData(Common.PropertyKey).getRawType())) {
				childId = saturateChildId(currentNode.getUserData(Common.ChildId))
						.toString();
			}
			stringBuilder.add(String.format("%s%s%s%s%s", down, startSymbol,
					currentNode.getUserData(Common.PropertyKey).getType(), childId, endSymbol));
		}

		return stringBuilder.toString();
	}

	private Integer saturateChildId(int childId) {
		return Math.min(childId, m_CommandLineValues.MaxChildId);
	}
}
