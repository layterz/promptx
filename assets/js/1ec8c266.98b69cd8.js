"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[182],{3905:(e,t,r)=>{r.d(t,{Zo:()=>u,kt:()=>h});var a=r(7294);function n(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function o(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,a)}return r}function s(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?o(Object(r),!0).forEach((function(t){n(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):o(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,a,n=function(e,t){if(null==e)return{};var r,a,n={},o=Object.keys(e);for(a=0;a<o.length;a++)r=o[a],t.indexOf(r)>=0||(n[r]=e[r]);return n}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)r=o[a],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}var l=a.createContext({}),p=function(e){var t=a.useContext(l),r=t;return e&&(r="function"==typeof e?e(t):s(s({},t),e)),r},u=function(e){var t=p(e.components);return a.createElement(l.Provider,{value:t},e.children)},c="mdxType",m={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},d=a.forwardRef((function(e,t){var r=e.components,n=e.mdxType,o=e.originalType,l=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),c=p(r),d=n,h=c["".concat(l,".").concat(d)]||c[d]||m[d]||o;return r?a.createElement(h,s(s({ref:t},u),{},{components:r})):a.createElement(h,s({ref:t},u))}));function h(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var o=r.length,s=new Array(o);s[0]=d;var i={};for(var l in t)hasOwnProperty.call(t,l)&&(i[l]=t[l]);i.originalType=e,i[c]="string"==typeof e?e:n,s[1]=i;for(var p=2;p<o;p++)s[p]=r[p];return a.createElement.apply(null,s)}return a.createElement.apply(null,r)}d.displayName="MDXCreateElement"},1244:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>l,contentTitle:()=>s,default:()=>m,frontMatter:()=>o,metadata:()=>i,toc:()=>p});var a=r(7462),n=(r(7294),r(3905));const o={},s="Evals",i={unversionedId:"guides/evals",id:"guides/evals",title:"Evals",description:"promptz provides simple validation via Pydantic models, but this doesn't help to evaluate the quality of the generated output. This is where evals comes in.",source:"@site/docs/guides/evals.md",sourceDirName:"guides",slug:"/guides/evals",permalink:"/promptz/guides/evals",draft:!1,tags:[],version:"current",frontMatter:{},sidebar:"tutorialSidebar",previous:{title:"Config",permalink:"/promptz/guides/config"},next:{title:"Examples",permalink:"/promptz/examples"}},l={},p=[],u={toc:p},c="wrapper";function m(e){let{components:t,...r}=e;return(0,n.kt)(c,(0,a.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h1",{id:"evals"},"Evals"),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},"promptz")," provides simple validation via ",(0,n.kt)("strong",{parentName:"p"},"Pydantic")," models, but this doesn't help to evaluate the quality of the generated output. This is where ",(0,n.kt)("strong",{parentName:"p"},"evals")," comes in."),(0,n.kt)("p",null,(0,n.kt)("strong",{parentName:"p"},"evals")," let you assess the quality of a prompt output by comparing it to some ideal reference and returning a score from 0 to 100. The score is calculated using the same logic used to query the embeddings space. Here's a simple example:"),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre",className:"language-python"},'from promptz import prompt, evaluate\n\nexpected = "The capital of France is Paris."\nactual = prompt("What is the capital of France?")\nscore = evaluate(actual, expected)\n>>> 100.0\n')),(0,n.kt)("p",null,"In this example the model returns the exact expected output, so the score is 100.0. Let's try a more realistic example:"),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre",className:"language-python"},'expected = "Batman is a superhero who protects Gotham City."\nactual = prompt("Write a description of Batman.")\nscore = evaluate(actual, expected)\n>>> 87.5\n')),(0,n.kt)("p",null,"In this case the model returns a similar but not identical response, so the score is 87.5. The score is calculated by comparing the embeddings of the expected and actual outputs. The closer the embeddings are, the higher the score."),(0,n.kt)("p",null,"You can do the same for structured data:"),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre",className:"language-python"},'expected = Character(name="Batman", description="Batman is a superhero who protects Gotham City.", age=32)\n\nactual = prompt("Generate a character profile for Batman.", output=Character)\nscore = evaluate(actual, expected)\n>>> 92.3\n')),(0,n.kt)("p",null,'This begs the question: what is a "good" score? There\'s not a single answer to this question, and will depend on the type of output you are evaluating. But, as a rough rule of thumb, a score of 80 or above is usually a good indicator that the output is of roughly the same format and anything over 90 usually means the content is similar.'),(0,n.kt)("p",null,"To evaluate prompts in production, a better way to think about this is as a form of anomaly detection. I.e. if a prompt typically produces output that scores ~85.0 compared to the expected output, then you could set a threshold of 80.0 and monitor the number of outputs that fall below that value."))}m.isMDXComponent=!0}}]);