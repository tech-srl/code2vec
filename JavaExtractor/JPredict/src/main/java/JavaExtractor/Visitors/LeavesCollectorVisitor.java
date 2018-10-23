package JavaExtractor.Visitors;

import java.util.ArrayList;
import java.util.List;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.TreeVisitor;

import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.Property;

public class LeavesCollectorVisitor extends TreeVisitor {
	ArrayList<Node> m_Leaves = new ArrayList<>(); 
	private int currentId = 1;

	@Override
	public void process(Node node) {
		if (node instanceof Comment) {
			return;
		}
		boolean isLeaf = false;
		boolean isGenericParent = isGenericParent(node);
		if (hasNoChildren(node) && isNotComment(node)) {
			if (!node.toString().isEmpty() && (!"null".equals(node.toString()) || (node instanceof NullLiteralExpr))) {
				m_Leaves.add(node);
				isLeaf = true;
			}
		}
		
		int childId = getChildId(node);
		node.setUserData(Common.ChildId, childId);
		Property property = new Property(node, isLeaf, isGenericParent, currentId++);
		node.setUserData(Common.PropertyKey, property);
	}

	private boolean isGenericParent(Node node) {
		return (node instanceof ClassOrInterfaceType) 
				&& ((ClassOrInterfaceType)node).getTypeArguments() != null 
				&& ((ClassOrInterfaceType)node).getTypeArguments().size() > 0;
	}

	private boolean hasNoChildren(Node node) {
		return node.getChildrenNodes().size() == 0;
	}
	
	private boolean isNotComment(Node node) {
		return !(node instanceof Comment) && !(node instanceof Statement);
	}
	
	public ArrayList<Node> getLeaves() {
		return m_Leaves;
	}
	
	private int getChildId(Node node) {
		Node parent = node.getParentNode();
		List<Node> parentsChildren = parent.getChildrenNodes();
		int childId = 0;
		for (Node child: parentsChildren) {
			if (child.getRange().equals(node.getRange())) {
				return childId;
			}
			childId++;
		}
		return childId;
	}
}
