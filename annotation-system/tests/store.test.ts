import { describe, expect, it } from 'vitest'
import { initialState, rois, slides } from '../src/data'
import { blindCandidateAdapter, canEditInitial, canEditUnion, exportRows, geometryToPercentRect, loadState, refineCandidate, rowsToCsv, STORAGE_KEY, viewerDragToGeometry } from '../src/store'

describe('protocol safeguards', () => {
  it('uses an allowlist so provenance and unknown future fields cannot leak to the blind DTO', () => {
    const union = rois.find(r => r.blindId)!
    const extended = {...union, sourceType:'full_agent' as const, sourceRunId:'should-never-leak', rank:99, members:[{secret:'no'}], futureProvenance:'no'}
    const dto = blindCandidateAdapter(extended)
    expect(dto).not.toHaveProperty('sourceType')
    expect(dto).not.toHaveProperty('sourceRunId')
    expect(dto).not.toHaveProperty('rank')
    expect(dto).not.toHaveProperty('members')
    expect(dto).not.toHaveProperty('futureProvenance')
    expect(dto).toEqual(expect.objectContaining({blindId:'B9837', caseId:'CASE-017'}))
  })
  it('refining a union candidate preserves the original geometry and increments revision', () => {
    const roi = rois.find(r => !r.sourceType)!
    const revised = refineCandidate(roi, {...roi.geometry, x: roi.geometry.x + 40})
    expect(revised.originalGeometry).toEqual(roi.geometry)
    expect(revised.geometry.x).toBe(roi.geometry.x + 40)
    expect(revised.revision).toBe(roi.revision + 1)
  })
  it('a frozen initial round rejects editing at the policy boundary', () => {
    expect(canEditInitial({...initialState,round:{...initialState.round,initial:'frozen'}})).toBe(false)
  })
  it('a frozen union round rejects adjudication edits at the policy boundary', () => {
    expect(canEditUnion({...initialState,round:{...initialState.round,union:'frozen'}})).toBe(false)
  })
  it('admin export preserves the deduplicated union-member provenance relationship', () => {
    const rows = exportRows(initialState)
    const merged = rows.find(r=>r.roi_id==='UNION-017-01')!
    expect(merged.member_count).toBe(3)
    expect(merged.members.map(member=>member.source_type)).toEqual(['expert_initial','fixed_topk','full_agent'])
    expect(merged.members[2]).toMatchObject({proposal_id:'ROI-A-017-04', model_version:'agent-evidence-1.2'})
  })
  it('normalizes a legacy localStorage state that has no unionMembers', () => {
    const { unionMembers: _unionMembers, ...legacy } = initialState
    localStorage.setItem(STORAGE_KEY, JSON.stringify(legacy))
    const restored = loadState()
    expect(restored.unionMembers).toEqual([])
    expect(restored.round).toEqual(initialState.round)
    expect(restored.rois).toHaveLength(initialState.rois.length)
    localStorage.removeItem(STORAGE_KEY)
  })
  it('maps level-0 geometry and reverse viewer drags using real slide dimensions', () => {
    const slide = slides.find(item=>item.id==='SLIDE-017-A')!
    expect(geometryToPercentRect({x:2390.4,y:4499.6,width:4780.8,height:2249.8},slide)).toEqual({left:'10%',top:'20%',width:'20%',height:'10%'})
    expect(viewerDragToGeometry({x:.8,y:.75},{x:.2,y:.25},slide)).toEqual({x:4781,y:5625,width:14342,height:11249})
  })
  it('writes nested members as one RFC4180-escaped CSV field', () => {
    const csv = rowsToCsv(exportRows(initialState))
    const [header, firstDataRow] = csv.split('\r\n')
    expect(header.split(',')).toHaveLength(Object.keys(exportRows(initialState)[0]).length)
    const columns = firstDataRow.match(/(?:^|,)("(?:[^"]|"")*"|[^,]*)/g) ?? []
    expect(columns).toHaveLength(header.split(',').length)
    const unionLine = csv.split('\r\n').find(line=>line.includes('UNION-017-01'))!
    expect(unionLine).toContain('"[{""proposal_id"":""ROI-E-017-01""')
  })
})
