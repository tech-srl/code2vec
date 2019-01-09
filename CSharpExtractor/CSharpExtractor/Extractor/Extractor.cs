using Extractor.Semantics;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;


namespace Extractor
{
    public class Extractor
    {
        //private SemanticModel semanticModel;
        public const string UpTreeChar = "^";
        public const string DownTreeChar = "_";
        public const string InternalDelimiter = "|";
        public const string MethodNameConst = "METHOD_NAME";
        // public const string UpTreeChar = InternalDelimiter;
        // public const string DownTreeChar = InternalDelimiter;
        public static SyntaxKind[] ParentTypeToAddChildId = new SyntaxKind[] { SyntaxKind.SimpleAssignmentExpression,
            SyntaxKind.ElementAccessExpression, SyntaxKind.SimpleMemberAccessExpression, SyntaxKind.InvocationExpression, SyntaxKind.BracketedArgumentList, SyntaxKind.ArgumentList};

        private ICollection<Variable> variables;

        public int LengthLimit { get; set; }
        public int WidthLimit { get; set; }
        public string Code { get; set; }
        public bool ShouldHash { get; set; }
        public int MaxContexts { get; set; }

        public Extractor(string code, Options opts)
		{
            LengthLimit = opts.MaxLength;
            WidthLimit = opts.MaxWidth;
            ShouldHash = !opts.NoHash;
            MaxContexts = opts.MaxContexts;
            Code = code;
		}


		StringBuilder builder = new StringBuilder();

		private string PathNodesToString(PathFinder.Path path)
		{
			builder.Clear();
            var nodeTypes = path.LeftSide;
			if (nodeTypes.Count() > 0)
			{
				builder.Append(nodeTypes.First().Kind());
                if (ParentTypeToAddChildId.Contains(nodeTypes.First().Parent.Kind()))
                {
                    builder.Append(GetTruncatedChildId(nodeTypes.First()));
                }
                foreach (var n in nodeTypes.Skip(1))
                {
                    builder.Append(UpTreeChar).Append(n.Kind());
                    if (ParentTypeToAddChildId.Contains(n.Parent.Kind()))
                    {
                        builder.Append(GetTruncatedChildId(n));
                    }
                }
				builder.Append(UpTreeChar);
			}
			builder.Append(path.Ancesstor.Kind());
            nodeTypes = path.RightSide;
			if (nodeTypes.Count() > 0)
			{
				builder.Append(DownTreeChar);
				builder.Append(nodeTypes.First().Kind());
                if (ParentTypeToAddChildId.Contains(nodeTypes.First().Parent.Kind()))
                {
                    builder.Append(GetTruncatedChildId(nodeTypes.First()));
                }
                foreach (var n in nodeTypes.Skip(1))
                {
                    builder.Append(DownTreeChar).Append(n.Kind());
                    if (ParentTypeToAddChildId.Contains(n.Parent.Kind()))
                    {
                        builder.Append(GetTruncatedChildId(n));
                    }
                }
				
			}
			return builder.ToString();
		}

        private int GetTruncatedChildId(SyntaxNode n)
        {
            var parent = n.Parent;
            int index = parent.ChildNodes().ToList().IndexOf(n);
            if (index > 3)
            {
                index = 3;
            }
            return index;
        }

        private string PathToString(PathFinder.Path path)
		{
			SyntaxNode ancesstor = path.Ancesstor;
			StringBuilder builder = new StringBuilder();
			builder.Append(path.Left.Text).Append(UpTreeChar);
			builder.Append(this.PathNodesToString(path));
			builder.Append(DownTreeChar).Append(path.Right.Text);
			return builder.ToString();
		}

        internal IEnumerable<PathFinder.Path> GetInternalPaths(Tree tree)
        {
            var finder = new PathFinder(tree, LengthLimit, WidthLimit);

            var allPairs = Utilities.ReservoirSample(Utilities.WeakConcat(Utilities.Choose2(variables),
                         variables.Select((arg) => new Tuple<Variable, Variable>(arg, arg))), MaxContexts);

            //iterate over variable-variable pairs
            foreach (Tuple<Variable, Variable> varPair in allPairs)
            {
                bool pathToSelf = varPair.Item1 == varPair.Item2;

                foreach (var rhs in varPair.Item2.Leaves)
                    foreach (var lhs in varPair.Item1.Leaves)
    				{
                        
                        if (lhs == rhs)
    						continue;

                        PathFinder.Path path = finder.FindPath(lhs, rhs, limited: true);

    					if (path == null)
    						continue;
                            
                        yield return path;
    				}
			}
		}

	    private string SplitNameUnlessEmpty(string original)
	    {
	        var subtokens = Utilities.SplitToSubtokens(original).Where(s => s.Length > 0);
            String name = String.Join(InternalDelimiter, subtokens);
	        if (name.Length == 0)
	        {
	            name = Utilities.NormalizeName(original);
	        }

	        if (String.IsNullOrWhiteSpace(name))
	        {
	            name = "SPACE";
	        }

	        if (String.IsNullOrEmpty(name))
	        {
	            name = "BLANK";
	        }
            if (original == Extractor.MethodNameConst)
            {
                name = original;
            }
	        return name;
        }


	    static readonly char[] removeFromComments = new char[] {' ', '/', '*', '{', '}'};

        public List<String> Extract()
		{
            var tree = new Tree(CSharpSyntaxTree.ParseText(Code).GetRoot());

            IEnumerable<MethodDeclarationSyntax> methods = tree.GetRoot().DescendantNodesAndSelf().OfType<MethodDeclarationSyntax>().ToList();

            List<String> results = new List<string>();

            foreach(var method in methods) {

                String methodName = method.Identifier.ValueText;
                Tree methodTree = new Tree(method);
                var subtokensMethodName = Utilities.SplitToSubtokens(methodName);
                var tokenToVar = new Dictionary<SyntaxToken, Variable>();
                this.variables = Variable.CreateFromMethod(methodTree).ToArray();

                foreach (var variable in variables)
                {
                    foreach (SyntaxToken token in variable.Leaves)
                    {
                        tokenToVar[token] = variable;
                    }
                }

                List<String> contexts = new List<String>();

                foreach (PathFinder.Path path in GetInternalPaths(methodTree))
                {
                    String pathString = SplitNameUnlessEmpty(tokenToVar[path.Left].Name)
                        + "," + MaybeHash(this.PathNodesToString(path))
                        + "," + SplitNameUnlessEmpty(tokenToVar[path.Right].Name);

                    Debug.WriteLine(path.Left.FullSpan+" "+tokenToVar[path.Left].Name+ "," +this.PathNodesToString(path)+ "," + tokenToVar[path.Right].Name+" "+path.Right.FullSpan);    
                    contexts.Add(pathString);
                }

                var commentNodes = tree.GetRoot().DescendantTrivia().Where(
                    node => node.IsKind(SyntaxKind.MultiLineCommentTrivia) || node.IsKind(SyntaxKind.SingleLineCommentTrivia) || node.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia));
                foreach (SyntaxTrivia trivia in commentNodes)
                {

                    string commentText = trivia.ToString().Trim(removeFromComments);

                    string normalizedTrivia = SplitNameUnlessEmpty(commentText);
                    var parts = normalizedTrivia.Split('|');
                    for (int i = 0; i < Math.Ceiling((double)parts.Length / (double)5); i++)
                    {
                        var batch = String.Join("|", parts.Skip(i * 5).Take(5));
                        contexts.Add(batch + "," + "COMMENT" + "," + batch);
                    }
                }
                results.Add(String.Join("|", subtokensMethodName) + " " + String.Join(" ", contexts));  
            }
            return results;
        }

        private string MaybeHash(string v)
        {
            if (this.ShouldHash)
            {
                return v.GetHashCode().ToString();
            } else
            {
                return v;
            }
        }
    }
}
