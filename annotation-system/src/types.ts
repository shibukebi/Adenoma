export type Role = 'pathologist' | 'adjudicator' | 'admin'
export type Stage = 'initial' | 'union'
export type Geometry = { x: number; y: number; width: number; height: number; polygon?: [number, number][] }
export type Slide = { id: string; widthLevel0: number; heightLevel0: number; mppX: number; mppY: number; imageUrl: string }
export type Relevance = 'none' | 'weak' | 'moderate' | 'strong'
export type Hgd = 'present' | 'absent' | 'not_assessable' | 'not_relevant'
export type Roi = {
  id: string; caseId: string; slideId: string; geometry: Geometry; physicalWidthUm: number; physicalHeightUm: number;
  evidence: string[]; relevance: Relevance; evaluable: boolean; supports: string[]; consistency: 'supports' | 'neutral' | 'contradicts'; hgd: Hgd;
  revision: number; status: 'draft' | 'submitted' | 'superseded'; createdAt: string; modifiedAt: string; originalGeometry?: Geometry;
  /** Present only for proposals and never returned by the blind candidate adapter. */
  sourceType?: 'expert_initial' | 'fixed_topk' | 'full_agent'; sourceRunId?: string; rank?: number; modelVersion?: string;
  /** Admin-only stable union identifier and the fixed blind label shown to adjudicators. */
  blindId?: string
}
export type UnionCandidateMember = {
  id: string; unionCandidateId: string; proposalId: string;
  sourceType: 'expert_initial' | 'fixed_topk' | 'full_agent'; sourceRunId?: string; rank?: number; modelVersion?: string
}
/** The only DTO an adjudicator API may return. It is deliberately an allowlist. */
export type BlindCandidate = {
  blindId: string; caseId: string; slideId: string; geometry: Geometry; physicalWidthUm: number; physicalHeightUm: number;
  evidence: string[]; relevance: Relevance; evaluable: boolean; supports: string[]; consistency: 'supports' | 'neutral' | 'contradicts'; hgd: Hgd;
  revision: number; status: 'draft' | 'submitted' | 'superseded'; createdAt: string; modifiedAt: string; originalGeometry?: Geometry
}
export type Case = { id: string; slideId: string; title: string; referenceDiagnosis: string; hgd: string; mpp: number; initialStatus: 'draft'|'submitted'|'frozen'; unionStatus: 'pending'|'active'|'submitted'; assignee: string }
export type Audit = { id: string; at: string; actor: string; action: string; detail: string }
export type AppState = { cases: Case[]; slides: Slide[]; rois: Roi[]; unionMembers: UnionCandidateMember[]; audits: Audit[]; round: { initial: 'draft'|'active'|'frozen'; union: 'draft'|'active'|'frozen' } }
