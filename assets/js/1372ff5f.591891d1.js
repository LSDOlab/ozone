"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[559],{3905:function(a,e,t){t.d(e,{Zo:function(){return o},kt:function(){return k}});var n=t(7294);function s(a,e,t){return e in a?Object.defineProperty(a,e,{value:t,enumerable:!0,configurable:!0,writable:!0}):a[e]=t,a}function m(a,e){var t=Object.keys(a);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(a);e&&(n=n.filter((function(e){return Object.getOwnPropertyDescriptor(a,e).enumerable}))),t.push.apply(t,n)}return t}function p(a){for(var e=1;e<arguments.length;e++){var t=null!=arguments[e]?arguments[e]:{};e%2?m(Object(t),!0).forEach((function(e){s(a,e,t[e])})):Object.getOwnPropertyDescriptors?Object.defineProperties(a,Object.getOwnPropertyDescriptors(t)):m(Object(t)).forEach((function(e){Object.defineProperty(a,e,Object.getOwnPropertyDescriptor(t,e))}))}return a}function r(a,e){if(null==a)return{};var t,n,s=function(a,e){if(null==a)return{};var t,n,s={},m=Object.keys(a);for(n=0;n<m.length;n++)t=m[n],e.indexOf(t)>=0||(s[t]=a[t]);return s}(a,e);if(Object.getOwnPropertySymbols){var m=Object.getOwnPropertySymbols(a);for(n=0;n<m.length;n++)t=m[n],e.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(a,t)&&(s[t]=a[t])}return s}var l=n.createContext({}),i=function(a){var e=n.useContext(l),t=e;return a&&(t="function"==typeof a?a(e):p(p({},e),a)),t},o=function(a){var e=i(a.components);return n.createElement(l.Provider,{value:e},a.children)},N={inlineCode:"code",wrapper:function(a){var e=a.children;return n.createElement(n.Fragment,{},e)}},c=n.forwardRef((function(a,e){var t=a.components,s=a.mdxType,m=a.originalType,l=a.parentName,o=r(a,["components","mdxType","originalType","parentName"]),c=i(t),k=s,h=c["".concat(l,".").concat(k)]||c[k]||N[k]||m;return t?n.createElement(h,p(p({ref:e},o),{},{components:t})):n.createElement(h,p({ref:e},o))}));function k(a,e){var t=arguments,s=e&&e.mdxType;if("string"==typeof a||s){var m=t.length,p=new Array(m);p[0]=c;var r={};for(var l in e)hasOwnProperty.call(e,l)&&(r[l]=e[l]);r.originalType=a,r.mdxType="string"==typeof a?a:s,p[1]=r;for(var i=2;i<m;i++)p[i]=t[i];return n.createElement.apply(null,p)}return n.createElement.apply(null,t)}c.displayName="MDXCreateElement"},6458:function(a,e,t){t.r(e),t.d(e,{frontMatter:function(){return r},contentTitle:function(){return l},metadata:function(){return i},toc:function(){return o},default:function(){return c}});var n=t(7462),s=t(3366),m=(t(7294),t(3905)),p=["components"],r={sidebar_position:1},l="Simple Example",i={unversionedId:"tutorial-basics/simple-example",id:"tutorial-basics/simple-example",title:"Simple Example",description:"To get an intuition on how to solve an ODE problem with ozone, lets walk through a simple example.",source:"@site/docs/tutorial-basics/simple-example.mdx",sourceDirName:"tutorial-basics",slug:"/tutorial-basics/simple-example",permalink:"/ozone/docs/tutorial-basics/simple-example",editUrl:"https://github.com/lsdolab/ozone/edit/main/website/docs/tutorial-basics/simple-example.mdx",tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"tutorialSidebar",previous:{title:"ODE Review",permalink:"/ozone/docs/tutorial-basics/ODEs"},next:{title:"Parameters and Coupling",permalink:"/ozone/docs/tutorial-basics/parameters-ode"}},o=[{value:"The Problem",id:"the-problem",children:[],level:2},{value:"Defining the ODE function.",id:"defining-the-ode-function",children:[],level:2},{value:"Creating the Solver Model",id:"creating-the-solver-model",children:[],level:2},{value:"Running the Model",id:"running-the-model",children:[],level:2}],N={toc:o};function c(a){var e=a.components,t=(0,s.Z)(a,p);return(0,m.kt)("wrapper",(0,n.Z)({},N,t,{components:e,mdxType:"MDXLayout"}),(0,m.kt)("h1",{id:"simple-example"},"Simple Example"),(0,m.kt)("p",null,"To get an intuition on how to solve an ODE problem with ",(0,m.kt)("inlineCode",{parentName:"p"},"ozone"),", lets walk through a simple example."),(0,m.kt)("h2",{id:"the-problem"},"The Problem"),(0,m.kt)("p",null,"Let's start by defining a simple ODE that we want to integrate,"),(0,m.kt)("div",{className:"math math-display"},(0,m.kt)("span",{parentName:"div",className:"katex-display"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML",display:"block"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mfrac",{parentName:"mrow"},(0,m.kt)("mrow",{parentName:"mfrac"},(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"\u2202"),(0,m.kt)("mi",{parentName:"mrow"},"y")),(0,m.kt)("mrow",{parentName:"mfrac"},(0,m.kt)("mi",{parentName:"mrow",mathvariant:"normal"},"\u2202"),(0,m.kt)("mi",{parentName:"mrow"},"t"))),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mo",{parentName:"mrow"},"\u2212"),(0,m.kt)("mi",{parentName:"mrow"},"y"),(0,m.kt)("mo",{parentName:"mrow",separator:"true"},","),(0,m.kt)("mspace",{parentName:"mrow",width:"1em"}),(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"y"),(0,m.kt)("mn",{parentName:"msub"},"0")),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mn",{parentName:"mrow"},"0.5.")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"\\frac{\\partial{y}}{\\partial{t}} = -y, \\quad y_0 = 0.5.")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"2.0574em",verticalAlign:"-0.686em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mopen nulldelimiter"}),(0,m.kt)("span",{parentName:"span",className:"mfrac"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"1.3714em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.314em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord",style:{marginRight:"0.05556em"}},"\u2202"),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"t")))),(0,m.kt)("span",{parentName:"span",style:{top:"-3.23em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3em"}}),(0,m.kt)("span",{parentName:"span",className:"frac-line",style:{borderBottomWidth:"0.04em"}})),(0,m.kt)("span",{parentName:"span",style:{top:"-3.677em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"3em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord",style:{marginRight:"0.05556em"}},"\u2202"),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"y"))))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.686em"}},(0,m.kt)("span",{parentName:"span"}))))),(0,m.kt)("span",{parentName:"span",className:"mclose nulldelimiter"})),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.7778em",verticalAlign:"-0.1944em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"\u2212"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"y"),(0,m.kt)("span",{parentName:"span",className:"mpunct"},","),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"1em"}}),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.1667em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"y"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.3011em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0359em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"0")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6444em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"0.5.")))))),(0,m.kt)("p",null,"We want to integrate it from ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"t"),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mn",{parentName:"mrow"},"0")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"t = 0")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6151em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"t"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6444em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"0")))))," to ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"t"),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mn",{parentName:"mrow"},"3")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"t = 3")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6151em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"t"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6444em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"3")))))," using 4th order Runge-Kutta (RK4). With the problem defined, let's see how to implement this in ",(0,m.kt)("inlineCode",{parentName:"p"},"ozone"),"."),(0,m.kt)("h2",{id:"defining-the-ode-function"},"Defining the ODE function."),(0,m.kt)("p",null,"We represent our ODE function ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mo",{parentName:"mrow"},"\u2212"),(0,m.kt)("mi",{parentName:"mrow"},"y")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"-y")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.7778em",verticalAlign:"-0.1944em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"\u2212"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"y")))))," as a ",(0,m.kt)("inlineCode",{parentName:"p"},"csdl")," ",(0,m.kt)("strong",{parentName:"p"},"Model()"),". For those unfamiliar, ",(0,m.kt)("inlineCode",{parentName:"p"},"csdl")," is an Embedded Domain Specific Language framework for modeling systems. For details, visit the csdl website."),(0,m.kt)("p",null,"Our csdl ",(0,m.kt)("inlineCode",{parentName:"p"},"Model")," takes in the state ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"y")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"y")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.625em",verticalAlign:"-0.1944em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"y")))))," as an input and outputs ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"d"),(0,m.kt)("mi",{parentName:"mrow"},"y"),(0,m.kt)("mi",{parentName:"mrow"},"d"),(0,m.kt)("mi",{parentName:"mrow"},"t")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"dydt")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.8889em",verticalAlign:"-0.1944em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"d"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"y"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"d"),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"t"))))),":"),(0,m.kt)("pre",null,(0,m.kt)("code",{parentName:"pre",className:"language-py"},"import csdl\n\nclass ODESystemModel(csdl.Model):\n    def initialize(self):\n        # Required every time for ODE systems or Profile Output systems\n        self.parameters.declare('num_nodes')\n\n    def define(self):\n        # Required every time for ODE systems or Profile Output systems\n        n = self.parameters['num_nodes']\n\n        # State y. All ODE states must have shape = (n, .. shape of state ...)\n        y = self.create_input('y', shape=n)\n\n        # Register output dy/dt = -y\n        self.register_output('dy_dt', -y)\n\n")),(0,m.kt)("h2",{id:"creating-the-solver-model"},"Creating the Solver Model"),(0,m.kt)("p",null,"We now have to create a separate ",(0,m.kt)("inlineCode",{parentName:"p"},"csdl")," model that creates our integrator."),(0,m.kt)("p",null,"First, let's define the inputs to the ODE: initial condition ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("msub",{parentName:"mrow"},(0,m.kt)("mi",{parentName:"msub"},"y"),(0,m.kt)("mn",{parentName:"msub"},"0")),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mn",{parentName:"mrow"},"0.5")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"y_0 = 0.5")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.625em",verticalAlign:"-0.1944em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},(0,m.kt)("span",{parentName:"span",className:"mord mathnormal",style:{marginRight:"0.03588em"}},"y"),(0,m.kt)("span",{parentName:"span",className:"msupsub"},(0,m.kt)("span",{parentName:"span",className:"vlist-t vlist-t2"},(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.3011em"}},(0,m.kt)("span",{parentName:"span",style:{top:"-2.55em",marginLeft:"-0.0359em",marginRight:"0.05em"}},(0,m.kt)("span",{parentName:"span",className:"pstrut",style:{height:"2.7em"}}),(0,m.kt)("span",{parentName:"span",className:"sizing reset-size6 size3 mtight"},(0,m.kt)("span",{parentName:"span",className:"mord mtight"},"0")))),(0,m.kt)("span",{parentName:"span",className:"vlist-s"},"\u200b")),(0,m.kt)("span",{parentName:"span",className:"vlist-r"},(0,m.kt)("span",{parentName:"span",className:"vlist",style:{height:"0.15em"}},(0,m.kt)("span",{parentName:"span"})))))),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6444em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"0.5")))))," and the timestep vector.\nThe timestep vector is a vector of stepsizes to integrate over. Let's use a timestep of 0.01s.\nThis means the timestep vector is ","[0.01, 0.01, ... 0.01, 0.01]"," with 30 elements to go from ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"t"),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mn",{parentName:"mrow"},"0")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"t = 0")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6151em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"t"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6444em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"0")))))," to ",(0,m.kt)("span",{parentName:"p",className:"math math-inline"},(0,m.kt)("span",{parentName:"span",className:"katex"},(0,m.kt)("span",{parentName:"span",className:"katex-mathml"},(0,m.kt)("math",{parentName:"span",xmlns:"http://www.w3.org/1998/Math/MathML"},(0,m.kt)("semantics",{parentName:"math"},(0,m.kt)("mrow",{parentName:"semantics"},(0,m.kt)("mi",{parentName:"mrow"},"t"),(0,m.kt)("mo",{parentName:"mrow"},"="),(0,m.kt)("mn",{parentName:"mrow"},"3")),(0,m.kt)("annotation",{parentName:"semantics",encoding:"application/x-tex"},"t = 3")))),(0,m.kt)("span",{parentName:"span",className:"katex-html","aria-hidden":"true"},(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6151em"}}),(0,m.kt)("span",{parentName:"span",className:"mord mathnormal"},"t"),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}}),(0,m.kt)("span",{parentName:"span",className:"mrel"},"="),(0,m.kt)("span",{parentName:"span",className:"mspace",style:{marginRight:"0.2778em"}})),(0,m.kt)("span",{parentName:"span",className:"base"},(0,m.kt)("span",{parentName:"span",className:"strut",style:{height:"0.6444em"}}),(0,m.kt)("span",{parentName:"span",className:"mord"},"3"))))),".\nBoth of these must be ",(0,m.kt)("inlineCode",{parentName:"p"},"csdl")," variables."),(0,m.kt)("p",null,"Next, we will create the ODE solver model. Import the ",(0,m.kt)("inlineCode",{parentName:"p"},"ODEProblem")," class from ",(0,m.kt)("inlineCode",{parentName:"p"},"ozone.api"),".\nInstantiate the class with three arguments:"),(0,m.kt)("ul",null,(0,m.kt)("li",{parentName:"ul"},"Numerical Method (RK4)"),(0,m.kt)("li",{parentName:"ul"},"Solution Approach (Timemarching)"),(0,m.kt)("li",{parentName:"ul"},"Number of timesteps (30)")),(0,m.kt)("p",null,"We then set the state, time vector and the ODE function. Finally, we create the solver model."),(0,m.kt)("pre",null,(0,m.kt)("code",{parentName:"pre",className:"language-py"},"import numpy as np\nfrom ozone.api import ODEProblem\n\n# The CSDL Model containing the ODE integrator\nclass RunModel(csdl.Model):\n\n    def define(self):\n        num_times = 30\n        dt = 0.1\n\n        # Create inputs to the ODE\n        # Initial condition for state\n        self.create_input('y_0', 0.5)\n\n        # Timestep vector\n        h_vec = np.ones(num_times)*dt\n        self.create_input('h', h_vec)\n\n        # Create ODEProblem class\n        ode_problem = ODEProblem('RK4', 'time-marching', num_times)\n        ode_problem.add_state('y', 'dy_dt', initial_condition_name='y_0', output='y_integrated')\n        ode_problem.add_times(step_vector='h')\n        ode_problem.set_ode_system(ODESystemModel)\n\n        # Create CSDL Model of solver\n        self.add(ode_problem.create_solver_model())\n")),(0,m.kt)("h2",{id:"running-the-model"},"Running the Model"),(0,m.kt)("p",null,"We can then run our ",(0,m.kt)("inlineCode",{parentName:"p"},"csdl")," model. Let's also compute the derivative of the outputs with respect to the inputs.\nThe output ",(0,m.kt)("inlineCode",{parentName:"p"},"y_integrated")," is a vector of the solved state at each timestep."),(0,m.kt)("pre",null,(0,m.kt)("code",{parentName:"pre",className:"language-py"},"import csdl_om\n\n# Simulator object:\nsim = csdl_om.Simulator(RunModel(), mode='rev')\n\n# Run and check derivatives\nsim.prob.run_model()\nprint('y integrated:', sim.prob['y_integrated'])\nsim.prob.check_totals(of=['y_integrated'], wrt=['y_0'], compact_print=True)\n")))}c.isMDXComponent=!0}}]);