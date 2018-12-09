using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Extractor
{
    public class Tree
    {
        public const string DummyClass = "IgnoreDummyClass";
        public const string DummyMethodName = "IgnoreDummyMethod";
        public const string DummyType = "IgnoreDummyType";
        internal static readonly SyntaxKind[] literals = { SyntaxKind.NumericLiteralToken, SyntaxKind.StringLiteralToken, SyntaxKind.CharacterLiteralToken };

        internal static readonly HashSet<SyntaxKind> identifiers = new HashSet<SyntaxKind>(new SyntaxKind[] { SyntaxKind.IdentifierToken }); //, SyntaxKind.VoidKeyword, SyntaxKind.StringKeyword });
        internal static readonly HashSet<SyntaxKind> keywords = new HashSet<SyntaxKind>(new SyntaxKind[] { SyntaxKind.RefKeyword, SyntaxKind.OutKeyword, SyntaxKind.ConstKeyword });
        internal static readonly HashSet<SyntaxKind> declarations = new HashSet<SyntaxKind>(new SyntaxKind[] { SyntaxKind.VariableDeclarator, SyntaxKind.Parameter, SyntaxKind.CatchDeclaration, SyntaxKind.ForEachStatement });
        internal static readonly HashSet<SyntaxKind> memberAccesses = new HashSet<SyntaxKind>(new SyntaxKind[] { SyntaxKind.SimpleMemberAccessExpression, SyntaxKind.PointerMemberAccessExpression });
        internal static readonly HashSet<SyntaxKind> scopeEnders = new HashSet<SyntaxKind>(
            new SyntaxKind[]{ SyntaxKind.Block, SyntaxKind.ForStatement, SyntaxKind.MethodDeclaration,
            SyntaxKind.ForEachStatement, SyntaxKind.CatchClause, SyntaxKind.SwitchSection, SyntaxKind.UsingStatement });

        internal static readonly HashSet<SyntaxKind> lambdaScopeStarters = new HashSet<SyntaxKind>(
            new SyntaxKind[]{ SyntaxKind.AnonymousMethodExpression,
            SyntaxKind.SimpleLambdaExpression, SyntaxKind.ParenthesizedLambdaExpression });

        public static bool IsScopeEnder(SyntaxNode node)
        {
            return Tree.scopeEnders.Contains(node.Kind());
        }

        class TreeBuilderWalker : CSharpSyntaxWalker
        {
            Dictionary<SyntaxNode, Node> nodes;
            HashSet<SyntaxNode> visitedNodes;
            List<SyntaxNode> Desc;
            List<SyntaxToken> Tokens;
            Dictionary<SyntaxToken, Leaf> tokens;

            internal TreeBuilderWalker(Dictionary<SyntaxNode, Node> nodes, Dictionary<SyntaxToken, Leaf> tokens)
            {
                visitedNodes = new HashSet<SyntaxNode>();
                this.nodes = nodes;
                this.tokens = tokens;
            }

            public override
            void Visit(SyntaxNode node)
            {
                visitedNodes.Add(node);

                base.Visit(node);

                visitedNodes.Remove(node);

                Desc = new List<SyntaxNode>();
                Tokens = new List<SyntaxToken>();
                foreach (var c in node.ChildNodes())
                {
                    if (!nodes.ContainsKey(c))
                    {
                        continue;
                    }
                    Desc.AddRange(nodes[c].Descendents);
                    Desc.Add(c);
                    Tokens.AddRange(nodes[c].Leaves);
                }
                foreach (var token in node.ChildTokens())
                {
                    if (Leaf.IsLeafToken(token))
                    {
                        tokens[token] = new Leaf(nodes, token);
                        Tokens.Add(token);
                    }
                }

                Node res = new Node(This: node,
                                       Ancestors: new HashSet<SyntaxNode>(visitedNodes),
                                       Descendents: Desc.ToArray(),
                                       Leaves: Tokens.ToArray(),
                                       Kind: node.Kind());
                nodes[node] = res;

            }
        }

        internal SyntaxNode GetRoot()
        {
            return tree;
        }

        SyntaxNode tree;
        internal Dictionary<SyntaxNode, Node> nodes = new Dictionary<SyntaxNode, Node>();
        internal Dictionary<SyntaxToken, Leaf> leaves = new Dictionary<SyntaxToken, Leaf>();

        public Tree(SyntaxNode syntaxTree)
        {
            this.tree = syntaxTree;

            /*if (this.tree.ChildNodes().ToList().Count() == 0)
            {
                this.tree = CSharpSyntaxTree.ParseText($"private {DummyType} {DummyMethodName}() {{ {code} }}");
            }*/
            new TreeBuilderWalker(nodes, leaves).Visit(this.tree);

            List<SyntaxTrivia> commentNodes = tree.DescendantTrivia().Where(
                node => node.IsKind(SyntaxKind.MultiLineCommentTrivia) || node.IsKind(SyntaxKind.SingleLineCommentTrivia)).ToList();

        }
    }

    public class Node
    {
        public Node(SyntaxNode This, HashSet<SyntaxNode> Ancestors, SyntaxNode[] Descendents,
                    SyntaxToken[] Leaves, SyntaxKind Kind)
        {
            this.This = This;
            this.Ancestors = Ancestors;
            this.Descendents = Descendents;
            this.AncestorsAndSelf = new HashSet<SyntaxNode>(Ancestors);
            this.AncestorsAndSelf.Add(This);
            this.Leaves = Leaves;
            this.Depth = Depth;
            this.Kind = Kind;
            this.KindName = Kind.ToString();
        }

        public SyntaxNode This { get; }

        public HashSet<SyntaxNode> Ancestors { get; }

        public HashSet<SyntaxNode> AncestorsAndSelf { get; }

        public SyntaxNode[] Descendents { get; }

        public SyntaxToken[] Leaves { get; }

        public SyntaxKind Kind { get; }

        public string KindName { get; }

        public int Depth { get; }

        public override bool Equals(object obj)
        {
            var item = obj as Node;

            if (item == null)
            {
                return false;
            }

            return this.This.Equals(item.This);
        }

        public override int GetHashCode()
        {
            return this.This.GetHashCode();
        }
    }

    public class Leaf
    {
        internal static bool IsLeafToken(SyntaxToken token)
        {
            if (token.Text.Equals("var") && token.IsKind(SyntaxKind.IdentifierToken)
                && token.Parent.IsKind(SyntaxKind.IdentifierName) && token.Parent.Parent.IsKind(SyntaxKind.VariableDeclaration)
                && token.Parent.Parent.Parent.IsKind(SyntaxKind.LocalDeclarationStatement))
            {
                return false;
            }

            if (token.ValueText == Tree.DummyMethodName || token.ValueText == Tree.DummyType)
            {
                return false;
            }

            return Tree.identifiers.Contains(token.Kind()) || Tree.literals.Contains(token.Kind()) || token.Parent.Kind() == SyntaxKind.PredefinedType;
        }

        public SyntaxToken token { get; }
        public SyntaxKind Kind { get; }
        public string KindName { get; }
        public string Text { get; set; }
        public bool IsConst { get; }
        public string VariableName { get; }

        public Leaf(Dictionary<SyntaxNode, Node> nodes, SyntaxToken token)
        {
            this.token = token;
            Kind = token.Kind();
            KindName = Kind.ToString();
            IsConst = !(Tree.identifiers.Contains(Kind) && Tree.declarations.Contains(token.Parent.Kind()));

            Text = token.ValueText;
            SyntaxNode node = token.Parent.Parent;
            SyntaxNode current = token.Parent;
            VariableName = Text;
        }
    }

    public class SyntaxViewer
    {
        private string ToDot(SyntaxTree tree)
        {
            List<SyntaxNode> nodes = tree.GetRoot().DescendantNodesAndSelf().ToList();
            SyntaxToken[] tokens = tree.GetRoot().DescendantTokens().ToArray();

            string[] tokenStrings = tokens.Select((arg) => arg.Kind().ToString() + "-" + arg.ToString()).ToArray();
            string[] nodeStrings = nodes.Select((arg) => arg.Kind().ToString()).ToArray();

            Dictionary<string, int> counts = new Dictionary<string, int>();
            Dictionary<int, string> nodeNames = new Dictionary<int, string>();
            IEnumerable<string> allItems = nodeStrings.Concat(tokenStrings);
            int i = 0;

            foreach (string name in allItems)
            {
                if (!counts.ContainsKey(name))
                    counts[name] = 0;
                counts[name] += 1;

                nodeNames[i] = name + counts[name].ToString();
                i++;
            }

            StringBuilder builder = new StringBuilder();
            builder.AppendLine("digraph G {");

            // vertexes
            for (i = 0; i < allItems.Count(); i++)
            {
                builder.AppendFormat("\"{0}\" ;\n", nodeNames[i]);
            }

            builder.AppendLine();

            // edges
            for (i = 1; i < nodes.Count(); i++)
            {
                builder.AppendFormat("\"{0}\"->\"{1}\" [];\n", nodeNames[nodes.IndexOf(nodes[i].Parent)], nodeNames[i]);
            }

            for (i = 0; i < tokens.Count(); i++)
            {
                builder.AppendFormat("\"{0}\"->\"{1}\" [];\n", nodeNames[nodes.IndexOf(tokens[i].Parent)], nodeNames[i + nodes.Count()]);
            }

            builder.AppendLine("}");
            return builder.ToString();
        }

        public SyntaxViewer(SyntaxTree tree, string path = "out.ong")
        {

            string dotData = ToDot(tree);
            
            File.WriteAllText("out.dot", dotData);
        }
    }
}
