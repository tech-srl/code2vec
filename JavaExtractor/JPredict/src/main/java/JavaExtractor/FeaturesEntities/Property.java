package JavaExtractor.FeaturesEntities;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.IntegerLiteralExpr;
import com.github.javaparser.ast.expr.UnaryExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;

import JavaExtractor.Common.Common;

public class Property {
	private String RawType;
	private String Type;
	private String Name;
	private String SplitName;
	private String Operator;
	public static final HashSet<String> NumericalKeepValues = Stream.of("0", "1", "32", "64")
			.collect(Collectors.toCollection(HashSet::new));

	public Property(Node node, boolean isLeaf, boolean isGenericParent, int id) {
		Class<?> nodeClass = node.getClass();
		RawType = Type = nodeClass.getSimpleName();
		if (node instanceof ClassOrInterfaceType && ((ClassOrInterfaceType) node).isBoxedType()) {
			Type = "PrimitiveType";
		}
		Operator = "";
		if (node instanceof BinaryExpr) {
			Operator = ((BinaryExpr) node).getOperator().toString();
		} else if (node instanceof UnaryExpr) {
			Operator = ((UnaryExpr) node).getOperator().toString();
		} else if (node instanceof AssignExpr) {
			Operator = ((AssignExpr) node).getOperator().toString();
		}
		if (Operator.length() > 0) {
			Type += ":" + Operator;
		}

		String nameToSplit = node.toString();
		if (isGenericParent) {
			nameToSplit = ((ClassOrInterfaceType) node).getName();
			if (isLeaf) {
				// if it is a generic parent which counts as a leaf, then when
				// it is participating in a path
				// as a parent, it should be GenericClass and not a simple
				// ClassOrInterfaceType.
				Type = "GenericClass";
			}
		}
		ArrayList<String> splitNameParts = Common.splitToSubtokens(nameToSplit);
		SplitName = splitNameParts.stream().collect(Collectors.joining(Common.internalSeparator));

		node.toString();
		Name = Common.normalizeName(node.toString(), Common.BlankWord);
		if (Name.length() > Common.c_MaxLabelLength) {
			Name = Name.substring(0, Common.c_MaxLabelLength);
		} else if (node instanceof ClassOrInterfaceType && ((ClassOrInterfaceType) node).isBoxedType()) {
			Name = ((ClassOrInterfaceType) node).toUnboxedType().toString();
		}

		if (Common.isMethod(node, Type)) {
			Name = SplitName = Common.methodName;
		}

		if (SplitName.length() == 0) {
			SplitName = Name;
			if (node instanceof IntegerLiteralExpr && !NumericalKeepValues.contains(SplitName)) {
				// This is a numeric literal, but not in our white list
				SplitName = "<NUM>";
			}
		}
	}

	public String getRawType() {
		return RawType;
	}

	public String getType() {
		return Type;
	}

	public String getName() {
		return Name;
	}
}
