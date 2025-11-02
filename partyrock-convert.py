import json

questions = """1. What permits and licenses do I need to start a food truck business?
2. How do I register my child for public school in a new district?
3. What documentation is needed to apply for a building permit for home renovation?
4. How do I file a small claims court case?
5. What steps are required to obtain a liquor license for my restaurant?
6. How do I appeal a property tax assessment?
7. What permits are needed to host a large public event?
8. How do I register a new business entity with the state?
9. What documentation is required for a passport renewal?
10. How do I apply for a zoning variance?
11. What steps are needed to become a licensed contractor?
12. How do I register my vehicle in a new state?
13. What permits are required for installing a swimming pool?
14. How do I obtain a permit for street parking in my neighborhood?
15. What documentation is needed to apply for a professional license?
16. How do I file a complaint against a licensed professional?
17. What permits are required for door-to-door sales?
18. How do I register to vote after moving to a new address?
19. What documentation is needed for a name change petition?
20. How do I apply for a permit to remove a protected tree?
21. What licenses are required to operate a bed and breakfast?
22. How do I file for unemployment benefits?
23. What permits are needed for residential fence installation?
24. How do I obtain a permit for a garage sale?
25. What documentation is required for a firearms permit?
26. How do I apply for a student loan deferment?
27. What permits are needed for street vending?
28. How do I register a trademark?
29. What documentation is needed for a disability parking permit?
30. How do I file a discrimination complaint?
31. What licenses are required for pest control services?
32. How do I apply for reduced school lunch programs?
33. What permits are needed for outdoor advertising?
34. How do I obtain a copy of property records?
35. What documentation is required for a fishing license?
36. How do I file an appeal for denied benefits?
37. What permits are needed for home-based business?
38. How do I register for selective service?
39. What documentation is needed for a hunting license?
40. How do I apply for a parade permit?
41. What licenses are required for private security services?
42. How do I file for workers' compensation?
43. What permits are needed for sidewalk repair?
44. How do I obtain a copy of vital records?
45. What documentation is required for a concealed carry permit?
46. How do I apply for food stamps?
47. What permits are needed for drone operation?
48. How do I register a domestic partnership?
49. What documentation is needed for a driver's license renewal?
50. How do I file a housing discrimination complaint?
51. What licenses are required for massage therapy practice?
52. How do I apply for Medicare benefits?
53. What permits are needed for temporary structures?
54. How do I obtain a permit for street closure?
55. What documentation is required for a real estate license?
56. How do I file for disability benefits?
57. What permits are needed for food handling?
58. How do I register a non-profit organization?
59. What documentation is needed for a business tax certificate?
60. How do I apply for a variance from building codes?
61. What licenses are required for child care facilities?
62. How do I file for bankruptcy protection?
63. What permits are needed for demolition work?
64. How do I obtain a copy of tax records?
65. What documentation is required for a teaching certificate?
66. How do I file an environmental complaint?
67. What permits are needed for special events?
68. How do I register a vehicle for commercial use?
69. What documentation is needed for a contractor's license?
70. How do I apply for Section 8 housing?
71. What licenses are required for food manufacturing?
72. How do I file for child support?
73. What permits are needed for sign installation?
74. How do I obtain a copy of court records?
75. What documentation is required for a pilot's license?
76. How do I file a consumer complaint?
77. What permits are needed for excavation work?
78. How do I register for Medicare Part D?
79. What documentation is needed for a plumbing license?
80. How do I apply for a historic preservation grant?
81. What licenses are required for alcohol sales?
82. How do I file for veterans benefits?
83. What permits are needed for outdoor seating?
84. How do I obtain a copy of marriage records?
85. What documentation is required for an electrician's license?
86. How do I file a workplace safety complaint?
87. What permits are needed for mobile food vending?
88. How do I register for social security benefits?
89. What documentation is needed for a pharmacy license?
90. How do I apply for a research grant?
91. What licenses are required for waste management?
92. How do I file for alimony?
93. What permits are needed for outdoor events?
94. How do I obtain a copy of birth records?
95. What documentation is required for a nursing license?
96. How do I file a civil rights complaint?
97. What permits are needed for construction work?
98. How do I register for unemployment insurance?
99. What documentation is needed for a dental license?
100. How do I apply for a business loan?
101. What licenses are required for transportation services?
102. How do I file for guardianship?
103. What permits are needed for landscaping work?
104. How do I obtain a copy of death records?
105. What documentation is required for a medical license?
106. How do I file a discrimination claim?
107. What permits are needed for outdoor advertising?
108. How do I register for disability insurance?
109. What documentation is needed for a veterinary license?
110. How do I apply for a research permit?
111. What licenses are required for security systems?
112. How do I file for custody rights?
113. What permits are needed for road work?
114. How do I obtain a copy of adoption records?
115. What documentation is required for an architect's license?
116. How do I file a workplace complaint?
117. What permits are needed for waste disposal?
118. How do I register for pension benefits?
119. What documentation is needed for an optometry license?
120. How do I apply for a building variance?
121. What licenses are required for private investigation?
122. How do I file for disability accommodation?
123. What permits are needed for tree removal?
124. How do I obtain a copy of divorce records?
125. What documentation is required for a counseling license?
126. How do I file an insurance claim?
127. What permits are needed for hazardous materials?
128. How do I register for retirement benefits?
129. What documentation is needed for a physical therapy license?
130. How do I apply for a zoning change?
131. What licenses are required for real estate appraisal?
132. How do I file for emergency assistance?
133. What permits are needed for public assembly?
134. How do I obtain a copy of military records?
135. What documentation is required for a psychology license?
136. How do I file a property dispute?
137. What permits are needed for water usage?
138. How do I register for health insurance?
139. What documentation is needed for a chiropractor's license?
140. How do I apply for a land use permit?
141. What licenses are required for commercial fishing?
142. How do I file for debt relief?
143. What permits are needed for public performance?
144. How do I obtain a copy of immigration records?
145. What documentation is required for a social work license?
146. How do I file a noise complaint?
147. What permits are needed for air quality?
148. How do I register for food assistance?
149. What documentation is needed for an engineering license?
150. How do I apply for a variance appeal?
151. What licenses are required for aviation services?
152. How do I file for bankruptcy discharge?
153. What permits are needed for public utilities?
154. How do I obtain a copy of business records?
155. What documentation is required for a pharmacy technician license?
156. How do I file a zoning complaint?
157. What permits are needed for street vendors?
158. How do I register for veterans services?
159. What documentation is needed for a massage therapy license?
160. How do I apply for a special use permit?
161. What licenses are required for commercial driving?
162. How do I file for medical leave?
163. What permits are needed for public markets?
164. How do I obtain a copy of educational records?
165. What documentation is required for a cosmetology license?
166. How do I file a building code violation?
167. What permits are needed for public transportation?
168. How do I register for disability services?
169. What documentation is needed for an accounting license?
170. How do I apply for a conditional use permit?
171. What licenses are required for food service?
172. How do I file for unemployment extension?
173. What permits are needed for public events?
174. How do I obtain a copy of employment records?
175. What documentation is required for a legal interpreter license?
176. How do I file a landlord complaint?
177. What permits are needed for public safety?
178. How do I register for social services?
179. What documentation is needed for a notary license?
180. How do I apply for a temporary permit?
181. What licenses are required for waste collection?
182. How do I file for workers rights?
183. What permits are needed for public health?
184. How do I obtain a copy of financial records?
185. What documentation is required for a private investigator license?
186. How do I file a tenant complaint?
187. What permits are needed for public works?
188. How do I register for emergency services?
189. What documentation is needed for a security guard license?
190. How do I apply for a variance request?
191. What licenses are required for pest control?
192. How do I file for family leave?
193. What permits are needed for public recreation?
194. How do I obtain a copy of insurance records?
195. What documentation is required for a real estate broker license?
196. How do I file a health code violation?
197. What permits are needed for public parking?
198. How do I register for housing assistance?
199. What documentation is needed for an insurance agent license?
200. How do I apply for a development permit?
201. What licenses are required for child care?
202. How do I file for disability rights?
203. What permits are needed for public gatherings?
204. How do I obtain a copy of medical records?
205. What documentation is required for a teacher certification?
206. How do I file a safety violation?
207. What permits are needed for public demonstrations?
208. How do I register for financial assistance?
209. What documentation is needed for a contractor license?
210. How do I apply for an occupancy permit?
211. What licenses are required for home inspection?
212. How do I file for civil rights?
213. What permits are needed for public art?
214. How do I obtain a copy of pension records?
215. What documentation is required for a nursing assistant license?
216. How do I file a discrimination complaint?
217. What permits are needed for public access?
218. How do I register for utility assistance?
219. What documentation is needed for a paralegal license?
220. How do I apply for a demolition permit?
221. What licenses are required for landscaping?
222. How do I file for equal opportunity?
223. What permits are needed for public space?
224. How do I obtain a copy of tax assessment records?
225. What documentation is required for a dental hygienist license?
226. How do I file a code violation?
227. What permits are needed for public facilities?
228. How do I register for rental assistance?
229. What documentation is needed for an appraiser license?
230. How do I apply for an excavation permit?
231. What licenses are required for tree service?
232. How do I file for consumer protection?
233. What permits are needed for public entertainment?
234. How do I obtain a copy of vehicle records?
235. What documentation is required for a physical therapist assistant license?
236. How do I file a workplace discrimination complaint?
237. What permits are needed for public storage?
238. How do I register for energy assistance?
239. What documentation is needed for a real estate agent license?
240. How do I apply for a construction permit?
241. What licenses are required for moving services?
242. How do I file for equal housing?
243. What permits are needed for public display?
244. How do I obtain a copy of property tax records?
245. What documentation is required for an occupational therapy license?
246. How do I file a fair housing complaint?
247. What permits are needed for public sales?
248. How do I register for medical assistance?
249. What documentation is needed for a home inspector license?
250. How do I apply for a sign permit?
251. What licenses are required for security services?
252. How do I file for fair labor?
253. What permits are needed for public markets?
254. How do I obtain a copy of licensing records?
255. What documentation is required for a speech therapist license?
256. How do I file an environmental violation?
257. What permits are needed for public auctions?
258. How do I register for transportation assistance?
259. What documentation is needed for a landscape architect license?
260. How do I apply for a fence permit?
261. What licenses are required for towing services?
262. How do I file for fair employment?
263. What permits are needed for public exhibitions?
264. How do I obtain a copy of zoning records?
265. What documentation is required for an optician license?
266. How do I file a civil rights violation?
267. What permits are needed for public performances?
268. How do I register for childcare assistance?
269. What documentation is needed for a land surveyor license?
270. How do I apply for a pool permit?
271. What licenses are required for cleaning services?
272. How do I file for fair wages?
273. What permits are needed for public meetings?
274. How do I obtain a copy of building records?
275. What documentation is required for an acupuncture license?
276. How do I file a housing violation?
277. What permits are needed for public festivals?
278. How do I register for legal assistance?
279. What documentation is needed for a private detective license?
280. How do I apply for a grading permit?
281. What licenses are required for storage services?
282. How do I file for fair credit?
283. What permits are needed for public ceremonies?
284. How do I obtain a copy of permit records?
285. What documentation is required for a massage therapist license?
286. How do I file a building violation?
287. What permits are needed for public parades?
288. How do I register for emergency housing?
289. What documentation is needed for an electrician license?
290. How do I apply for a parking permit?
291. What licenses are required for delivery services?
292. How do I file for fair trade?
293. What permits are needed for public protests?
294. How do I obtain a copy of inspection records?
295. What documentation is required for a plumber license?
296. How do I file a safety complaint?
297. What permits are needed for public fundraising?
298. How do I register for emergency food assistance?
299. What documentation is needed for an HVAC license?
300. How do I apply for a special events permit?"""

# Category mapping based on keywords
def categorize_question(question):
    q_lower = question.lower()
    
    if 'permit' in q_lower and 'food' in q_lower:
        return 'permits'
    elif 'license' in q_lower and any(word in q_lower for word in ['contractor', 'professional', 'massage', 'real estate', 'pharmacy', 'nursing', 'medical', 'dental', 'veterinary', 'plumbing', 'electrician', 'hvac', 'pilot', 'commercial driving', 'pest control', 'security', 'private investigation', 'cosmetology', 'acupuncture', 'optometry', 'chiropractor', 'psychology', 'social work', 'counseling', 'physical therapy', 'occupational therapy', 'speech therapist', 'landscape architect', 'land surveyor', 'accounting', 'engineering', 'architect', 'home inspector', 'appraiser', 'broker', 'notary', 'paralegal', 'interpreter']):
        return 'occupational licenses'
    elif 'license' in q_lower:
        return 'licenses'
    elif 'permit' in q_lower:
        return 'permits'
    elif 'register' in q_lower and any(word in q_lower for word in ['business', 'non-profit', 'trademark']):
        return 'small business'
    elif 'register' in q_lower and any(word in q_lower for word in ['vote', 'vehicle', 'selective service', 'partnership']):
        return 'registrations'
    elif 'register' in q_lower and any(word in q_lower for word in ['medicare', 'social security', 'unemployment', 'disability', 'veterans', 'pension', 'retirement', 'health insurance', 'food assistance', 'housing', 'energy', 'medical', 'childcare', 'emergency', 'financial', 'rental', 'utility', 'transportation', 'legal']):
        return 'benefits'
    elif 'file' in q_lower and any(word in q_lower for word in ['complaint', 'violation']):
        return 'complaints'
    elif 'file' in q_lower and any(word in q_lower for word in ['appeal', 'bankruptcy', 'dispute', 'claim', 'small claims']):
        return 'appeals'
    elif 'file' in q_lower and any(word in q_lower for word in ['unemployment', 'workers compensation', 'discrimination', 'workplace', 'fair labor', 'fair employment', 'fair wages', 'workers rights', 'family leave', 'medical leave']):
        return 'labor laws and unemployment'
    elif 'file' in q_lower and 'veterans' in q_lower:
        return 'military and veterans'
    elif 'file' in q_lower and any(word in q_lower for word in ['guardianship', 'custody', 'child support', 'alimony']):
        return 'laws and legal issues'
    elif 'file' in q_lower and 'disability' in q_lower:
        return 'disability services'
    elif 'appeal' in q_lower and 'tax' in q_lower:
        return 'taxes'
    elif 'obtain a copy' in q_lower or 'copy of' in q_lower:
        return 'inquiries'
    elif 'apply' in q_lower and any(word in q_lower for word in ['food stamps', 'section 8', 'medicare', 'school lunch', 'grant', 'loan']):
        return 'aids'
    elif 'apply' in q_lower and 'student loan' in q_lower:
        return 'education'
    elif 'variance' in q_lower or 'zoning' in q_lower:
        return 'permits'
    elif 'documentation' in q_lower and 'passport' in q_lower:
        return 'inquiries'
    elif 'school' in q_lower or 'teaching' in q_lower or 'teacher' in q_lower or 'educational' in q_lower:
        return 'education'
    elif 'business' in q_lower:
        return 'small business'
    elif 'housing' in q_lower or 'landlord' in q_lower or 'tenant' in q_lower:
        return 'laws and legal issues'
    elif 'disability' in q_lower:
        return 'disability services'
    elif 'veterans' in q_lower or 'military' in q_lower:
        return 'military and veterans'
    elif 'tax' in q_lower:
        return 'taxes'
    else:
        return 'general'

# Parse questions and create JSONL
lines = questions.strip().split('\n')
jsonl_output = []

for line in lines:
    # Remove number prefix
    question = line.split('. ', 1)[1] if '. ' in line else line
    question = question.strip()
    
    category = categorize_question(question)
    
    entry = {
        "instruction": question,
        "context": "",
        "response": "",
        "category": category
    }
    jsonl_output.append(json.dumps(entry))

# Print JSONL
output = '\n'.join(jsonl_output)
print(output)

# Also save to file
with open('eval_questions.jsonl', 'w') as f:
    f.write(output)

print(f"\n\n✓ Created eval_questions.jsonl with {len(jsonl_output)} questions")