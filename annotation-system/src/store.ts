import { initialState } from './data'
import type { AppState, BlindCandidate, Geometry, Roi, Slide, UnionCandidateMember } from './types'
export const STORAGE_KEY = 'pathology-evidence-annotation-mvp-v1'
/** JSON cloning avoids structuredClone, which is unavailable in Firefox 86. */
export function cloneInitialState(): AppState { return JSON.parse(JSON.stringify(initialState)) as AppState }
/**
 * Keeps saved demo work while supplying fields added in later MVP releases.
 * In particular, v0 states have no unionMembers, so they remain usable in Admin/QC.
 */
export function normalizeState(saved: Partial<AppState>): AppState {
  const fallback = cloneInitialState()
  return {
    ...fallback,
    ...saved,
    cases: Array.isArray(saved.cases) ? saved.cases : fallback.cases,
    slides: Array.isArray(saved.slides) ? saved.slides : fallback.slides,
    rois: Array.isArray(saved.rois) ? saved.rois : fallback.rois,
    unionMembers: Array.isArray(saved.unionMembers) ? saved.unionMembers : [],
    audits: Array.isArray(saved.audits) ? saved.audits : fallback.audits,
    round: {...fallback.round, ...(saved.round ?? {})}
  }
}
export function loadState(): AppState { try { const saved=localStorage.getItem(STORAGE_KEY); return saved ? normalizeState(JSON.parse(saved) as Partial<AppState>) : cloneInitialState() } catch { return cloneInitialState() } }
export function persistState(value: AppState) { localStorage.setItem(STORAGE_KEY, JSON.stringify(value)) }
/** Compatibility-safe, collision-resistant enough identifier for local audit records. */
export function createLocalId(prefix='audit'): string { return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2,10)}` }
/** Explicit allowlist: provenance, union member relations and any future fields cannot leak by default. */
export function blindCandidateAdapter(roi: Roi): BlindCandidate {
  return { blindId: roi.blindId ?? roi.id, caseId: roi.caseId, slideId: roi.slideId, geometry: roi.geometry, physicalWidthUm: roi.physicalWidthUm, physicalHeightUm: roi.physicalHeightUm, evidence: roi.evidence, relevance: roi.relevance, evaluable: roi.evaluable, supports: roi.supports, consistency: roi.consistency, hgd: roi.hgd, revision: roi.revision, status: roi.status, createdAt: roi.createdAt, modifiedAt: roi.modifiedAt, ...(roi.originalGeometry ? { originalGeometry: roi.originalGeometry } : {}) }
}
export function refineCandidate(roi: Roi, geometry: Roi['geometry']): Roi { return {...roi, originalGeometry: roi.originalGeometry ?? roi.geometry, geometry, revision:roi.revision + 1, modifiedAt:new Date().toISOString()} }
export function canEditInitial(state:AppState) { return state.round.initial !== 'frozen' }
export function canEditUnion(state:AppState) { return state.round.union !== 'frozen' }
function serialiseMembers(members: UnionCandidateMember[]) { return members.map(member=>({proposal_id:member.proposalId,source_type:member.sourceType,source_run_id:member.sourceRunId ?? null,rank:member.rank ?? null,model_version:member.modelVersion ?? null})) }
/** Admin-only export joins union candidates to preserved proposal provenance for unblinding. */
export function exportRows(state: AppState) {
  return state.rois.map(r=>{
    const members = r.blindId ? state.unionMembers.filter(m=>m.unionCandidateId===r.id) : []
    return {roi_id:r.id,blind_id:r.blindId ?? null,case_id:r.caseId,geometry:r.geometry,evidence:r.evidence,relevance:r.relevance,evaluable:r.evaluable,supports:r.supports,revision:r.revision,status:r.status,source_type:r.sourceType ?? 'union_candidate',member_count:members.length,members:serialiseMembers(members)}
  })
}
export function unionMemberCount(state: AppState, unionCandidateId: string) { return state.unionMembers.filter(member=>member.unionCandidateId===unionCandidateId).length }
const clamp = (value:number, minimum:number, maximum:number) => Math.min(Math.max(value, minimum), maximum)
/** Maps level-0 WSI coordinates to overlay percentages for the viewport. */
export function geometryToPercentRect(geometry: Geometry, slide: Slide) { return { left: `${(geometry.x / slide.widthLevel0) * 100}%`, top: `${(geometry.y / slide.heightLevel0) * 100}%`, width: `${(geometry.width / slide.widthLevel0) * 100}%`, height: `${(geometry.height / slide.heightLevel0) * 100}%` } }
/** Converts two viewer-relative points to a clamped level-0 rectangle, including reverse drags. */
export function viewerDragToGeometry(start:{x:number;y:number}, end:{x:number;y:number}, slide: Slide): Geometry {
  const x1=clamp(Math.min(start.x,end.x),0,1), x2=clamp(Math.max(start.x,end.x),0,1)
  const y1=clamp(Math.min(start.y,end.y),0,1), y2=clamp(Math.max(start.y,end.y),0,1)
  return {x:Math.round(x1*slide.widthLevel0),y:Math.round(y1*slide.heightLevel0),width:Math.round((x2-x1)*slide.widthLevel0),height:Math.round((y2-y1)*slide.heightLevel0)}
}
export function csvEscape(value: unknown) { const text=typeof value==='object' && value!==null ? JSON.stringify(value) : String(value ?? ''); return `"${text.replace(/"/g,'""')}"` }
/** RFC4180-safe export: every field is quoted, nested data is one JSON column. */
export function rowsToCsv(rows: Record<string, unknown>[]) { if(!rows.length)return ''; const header=Object.keys(rows[0]); return [header.join(','),...rows.map(row=>header.map(key=>csvEscape(row[key])).join(','))].join('\r\n') }
