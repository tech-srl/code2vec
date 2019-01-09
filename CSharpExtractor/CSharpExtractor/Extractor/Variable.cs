using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Extractor
{
	namespace Semantics
	{
		public class Variable
		{
			Tree tree;

			public string Name { get; }
			private HashSet<SyntaxToken> leaves;
			public HashSet<SyntaxToken> Leaves
			{
				get
				{
					return leaves;
				}
			}

			private Nullable<bool> constant;
			public bool Const
			{
				get
				{
					return constant.Value;
				}
			}


			private Variable(string name, SyntaxToken[] leaves, Tree tree)
			{
				this.tree = tree;
				this.Name = name;
				this.leaves = new HashSet<SyntaxToken>(leaves);


				constant = true;
				foreach (var leaf in leaves)
				{
					if (!tree.leaves[leaf].IsConst)
					{
						constant = false;
						// If not constant the it is a decleration token
						break;
					}
				}
			}

			public override int GetHashCode()
			{
				return this.Name.GetHashCode();
			}

			public bool IsLiteral()
			{
				return Tree.literals.Contains(tree.leaves[Leaves.First()].Kind);
			}

            internal static Boolean isMethodName(SyntaxToken token)
            {
                return token.Parent.IsKind(Microsoft.CodeAnalysis.CSharp.SyntaxKind.MethodDeclaration) 
                    && token.IsKind(Microsoft.CodeAnalysis.CSharp.SyntaxKind.IdentifierToken);
            }

			// Create a variable for each variable in scope from tokens while splitting identically named but differently scoped vars.
			internal static IEnumerable<Variable> CreateFromMethod(Tree methodTree)
			{
			    var root = methodTree.nodes[methodTree.GetRoot()];
				var leaves = root.Leaves.ToArray();
				Dictionary<SyntaxToken, string> tokenToName = new Dictionary<SyntaxToken, string>();
				Dictionary<string, List<SyntaxToken>> nameToTokens = new Dictionary<string, List<SyntaxToken>>();
				foreach (SyntaxToken token in root.Leaves)
				{
					string name = methodTree.leaves[token].VariableName;
                    if (isMethodName(token))
                    {
                        name = Extractor.MethodNameConst;
                    }
                    tokenToName[token] = name;
					if (!nameToTokens.ContainsKey(name))
						nameToTokens[name] = new List<SyntaxToken>();
					nameToTokens[name].Add(token);
				}

                List<Variable> results = new List<Variable>();

                foreach (SyntaxToken leaf in leaves)
				{
					string name = tokenToName[leaf];
					SyntaxToken[] syntaxTokens = nameToTokens[name].ToArray();
                    var v = new Variable(name, syntaxTokens, methodTree);

                    //check if exists
                    var matches = results.Where(p => p.Name == name).ToList();
                    bool alreadyExists = (matches.Count != 0);
                    if (!alreadyExists)
                    {
                        results.Add(v);
                    }
                }

                return results;
			}
		}
	}
}
