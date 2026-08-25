import type { AppState, Case, Roi, Slide, UnionCandidateMember } from './types'

export const VOCABULARY = [
  ['SSL / HP architecture', ['surface_serration','crypt_serration','basal_crypt_dilation','horizontal_crypt_growth','L_shaped_crypt','crypt_branching','crypt_distortion','mature_surface_epithelium']],
  ['TSA', ['villiform_architecture','ectopic_crypt_formation','eosinophilic_cytoplasm','pencillate_nuclei','slit_like_serration']],
  ['Conventional adenoma', ['tubular_architecture','villous_architecture','pseudostratification','hyperchromasia','loss_of_maturation']]
] as const
export const LABELS: Record<string,string> = { surface_serration:'表面锯齿', crypt_serration:'隐窝锯齿', basal_crypt_dilation:'隐窝基底扩张', horizontal_crypt_growth:'水平隐窝生长', L_shaped_crypt:'L 形隐窝', crypt_branching:'隐窝分支', crypt_distortion:'隐窝变形', mature_surface_epithelium:'成熟表面上皮', villiform_architecture:'绒毛状结构', ectopic_crypt_formation:'异位隐窝形成', eosinophilic_cytoplasm:'嗜酸性胞质', pencillate_nuclei:'栅栏样细胞核', slit_like_serration:'裂隙样锯齿', tubular_architecture:'管状结构', villous_architecture:'绒毛状结构', pseudostratification:'假复层', hyperchromasia:'深染', loss_of_maturation:'成熟缺失' }
export const cases: Case[] = [
  {id:'CASE-017',slideId:'SLIDE-017-A',title:'升结肠息肉',referenceDiagnosis:'SSL',hgd:'absent',mpp:0.2517,initialStatus:'draft',unionStatus:'active',assignee:'P001'},
  {id:'CASE-021',slideId:'SLIDE-021-A',title:'横结肠息肉',referenceDiagnosis:'TA',hgd:'absent',mpp:0.50,initialStatus:'submitted',unionStatus:'active',assignee:'P001'},
  {id:'CASE-039',slideId:'SLIDE-039-A',title:'乙状结肠息肉',referenceDiagnosis:'TSA',hgd:'present',mpp:0.25,initialStatus:'draft',unionStatus:'pending',assignee:'P001'},
  {id:'CASE-044',slideId:'SLIDE-044-A',title:'直肠病变',referenceDiagnosis:'HP',hgd:'absent',mpp:0.50,initialStatus:'submitted',unionStatus:'submitted',assignee:'P001'}
]
export const slides: Slide[] = [
  {id:'SLIDE-017-A',widthLevel0:23904,heightLevel0:22498,mppX:0.2517,mppY:0.2517,imageUrl:'/demo-slide.jpg'},
  {id:'SLIDE-021-A',widthLevel0:21840,heightLevel0:19820,mppX:0.50,mppY:0.50,imageUrl:'/demo-slide.jpg'},
  {id:'SLIDE-039-A',widthLevel0:28560,heightLevel0:21240,mppX:0.25,mppY:0.25,imageUrl:'/demo-slide.jpg'},
  {id:'SLIDE-044-A',widthLevel0:20480,heightLevel0:18640,mppX:0.50,mppY:0.50,imageUrl:'/demo-slide.jpg'}
]
const now = '2026-08-13T09:32:00.000Z'
export const rois: Roi[] = [
  {id:'ROI-E-017-01',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:2100,y:1650,width:930,height:720},physicalWidthUm:234,physicalHeightUm:181,evidence:['basal_crypt_dilation','horizontal_crypt_growth'],relevance:'strong',evaluable:true,supports:['SSL'],consistency:'supports',hgd:'not_relevant',revision:1,status:'draft',createdAt:now,modifiedAt:now,sourceType:'expert_initial',sourceRunId:'initial-v1',rank:1},
  {id:'ROI-F-017-01',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:2105,y:1660,width:920,height:710},physicalWidthUm:232,physicalHeightUm:179,evidence:[],relevance:'none',evaluable:true,supports:[],consistency:'neutral',hgd:'not_relevant',revision:1,status:'submitted',createdAt:now,modifiedAt:now,sourceType:'fixed_topk',sourceRunId:'fixed-topk-v3',rank:1,modelVersion:'fixed-grid-2026.08'},
  {id:'ROI-A-017-04',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:2090,y:1645,width:940,height:725},physicalWidthUm:237,physicalHeightUm:182,evidence:[],relevance:'none',evaluable:true,supports:[],consistency:'neutral',hgd:'not_relevant',revision:1,status:'submitted',createdAt:now,modifiedAt:now,sourceType:'full_agent',sourceRunId:'agent-run-017',rank:4,modelVersion:'agent-evidence-1.2'},
  {id:'ROI-F-017-02',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:4650,y:2750,width:820,height:670},physicalWidthUm:206,physicalHeightUm:169,evidence:[],relevance:'none',evaluable:true,supports:[],consistency:'neutral',hgd:'not_relevant',revision:1,status:'submitted',createdAt:now,modifiedAt:now,sourceType:'fixed_topk',sourceRunId:'fixed-topk-v3',rank:2,modelVersion:'fixed-grid-2026.08'},
  {id:'ROI-A-017-02',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:3150,y:4900,width:900,height:700},physicalWidthUm:227,physicalHeightUm:176,evidence:[],relevance:'none',evaluable:true,supports:[],consistency:'neutral',hgd:'not_relevant',revision:1,status:'submitted',createdAt:now,modifiedAt:now,sourceType:'full_agent',sourceRunId:'agent-run-017',rank:2,modelVersion:'agent-evidence-1.2'},
  {id:'UNION-017-01',blindId:'B9837',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:2100,y:1650,width:930,height:720},physicalWidthUm:234,physicalHeightUm:181,evidence:['surface_serration'],relevance:'moderate',evaluable:true,supports:['SSL'],consistency:'supports',hgd:'absent',revision:1,status:'draft',createdAt:now,modifiedAt:now},
  {id:'UNION-017-02',blindId:'B2731',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:6200,y:3600,width:780,height:640},physicalWidthUm:196,physicalHeightUm:161,evidence:[],relevance:'none',evaluable:true,supports:['non-specific'],consistency:'neutral',hgd:'not_relevant',revision:1,status:'draft',createdAt:now,modifiedAt:now},
  {id:'UNION-017-03',blindId:'B5412',caseId:'CASE-017',slideId:'SLIDE-017-A',geometry:{x:3150,y:4900,width:900,height:700},physicalWidthUm:227,physicalHeightUm:176,evidence:['crypt_distortion'],relevance:'weak',evaluable:true,supports:['SSL'],consistency:'supports',hgd:'absent',revision:1,status:'draft',createdAt:now,modifiedAt:now}
]
export const unionMembers: UnionCandidateMember[] = [
  {id:'U-M-001',unionCandidateId:'UNION-017-01',proposalId:'ROI-E-017-01',sourceType:'expert_initial',sourceRunId:'initial-v1',rank:1},
  {id:'U-M-002',unionCandidateId:'UNION-017-01',proposalId:'ROI-F-017-01',sourceType:'fixed_topk',sourceRunId:'fixed-topk-v3',rank:1,modelVersion:'fixed-grid-2026.08'},
  {id:'U-M-003',unionCandidateId:'UNION-017-01',proposalId:'ROI-A-017-04',sourceType:'full_agent',sourceRunId:'agent-run-017',rank:4,modelVersion:'agent-evidence-1.2'},
  {id:'U-M-004',unionCandidateId:'UNION-017-02',proposalId:'ROI-F-017-02',sourceType:'fixed_topk',sourceRunId:'fixed-topk-v3',rank:2,modelVersion:'fixed-grid-2026.08'},
  {id:'U-M-005',unionCandidateId:'UNION-017-03',proposalId:'ROI-A-017-02',sourceType:'full_agent',sourceRunId:'agent-run-017',rank:2,modelVersion:'agent-evidence-1.2'}
]
export const initialState: AppState = { cases, slides, rois, unionMembers, audits:[{id:'audit-1',at:now,actor:'系统',action:'生成盲法队列',detail:'Union v1 固定种子：study-2026-evidence-v1'}],round:{initial:'active',union:'active'} }
