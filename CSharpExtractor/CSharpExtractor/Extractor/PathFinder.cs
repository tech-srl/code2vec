using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Extractor
{

    internal class PathFinder
	{
		internal class Path
		{
			public SyntaxToken Left { get; }
			public List<SyntaxNode> LeftSide { get; }
			public SyntaxNode Ancesstor { get; }
			public List<SyntaxNode> RightSide { get; }
			public SyntaxToken Right { get; }

			public Path(SyntaxToken left, IEnumerable<SyntaxNode> leftSide, SyntaxNode ancesstor, 
			            IEnumerable<SyntaxNode> rightSide, SyntaxToken right)
			{
				this.Left = left;
				this.LeftSide = leftSide.ToList();
				this.Ancesstor = ancesstor;
				this.RightSide = rightSide.ToList();
				this.Right = right;
			}
		}

		public int Length { get; }
		public int Width { get; }

		Tree tree;

		public PathFinder(Tree tree, int length = 7, int width = 4)
		{
			if (length < 1 || width < 1)
				throw new ArgumentException("Width and Length params must be positive.");

			Length = length;
			Width = width;
			this.tree = tree;
		}

		private int GetDepth(SyntaxNode n)
		{
            int depth = 0;
			while(n.Parent != null)
            {
                n = n.Parent;
                depth++;
            }
            return depth;
		}

		public SyntaxNode FirstAncestor(SyntaxNode l, SyntaxNode r)
		{
			if (l.Equals(r))
				return l;

			if (GetDepth(l) >= GetDepth(r))
			{
				l = l.Parent;
			}
			else
			{
				r = r.Parent;
			}
			return FirstAncestor(l, r);
		}

		private IEnumerable<SyntaxNode> CollectPathToParent(SyntaxNode start, SyntaxNode parent)
		{
			while (!start.Equals(parent))
			{
				yield return start;
				start = start.Parent;
			}
		}

		internal Path FindPath(SyntaxToken l, SyntaxToken r, bool limited = true)
		{
			SyntaxNode p = FirstAncestor(l.Parent, r.Parent);

			// + 2 for the distance of the leafs themselves
			if (GetDepth(r.Parent) + GetDepth(l.Parent) - 2 * GetDepth(p) + 2 > Length)
			{
				return null;
			}

			var leftSide = CollectPathToParent(l.Parent, p);
			var rightSide = CollectPathToParent(r.Parent, p);
			rightSide = rightSide.Reverse();

			List<SyntaxNode> widthCheck = p.ChildNodes().ToList();
			if (limited && leftSide.Count() != 0
			    && rightSide.Count() != 0)
			{
				int indexOfLeft = widthCheck.IndexOf(leftSide.Last());
				int indexOfRight = widthCheck.IndexOf(rightSide.First());
				if (Math.Abs(indexOfLeft - indexOfRight) >= Width)
				{
					return null;
				}
			}

			return new Path(l, leftSide, p, rightSide, r);
		}
	}
}
