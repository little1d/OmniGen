export type ReviewStatus = 'pending' | 'approved' | 'needs_changes' | 'rejected'

export interface Review {
  status: ReviewStatus
  issues: string[]
  note: string
  caption: string | null
  updated_at: string | null
}

export interface MoleculeSummary {
  id: string
  display_key: string
  source: string
  source_label: string
  source_id?: string
  iupac_name?: string
  formula?: string
  smiles?: string
  caption_chars: number
  review: Review
}

export interface MoleculeDetail extends MoleculeSummary {
  caption: string
  effective_caption: string
  prompt?: string
  prompt_label?: string
  sdf?: string
  png?: string
  facts: Record<string, unknown>
  model?: string
  usage?: Record<string, unknown>
  provenance: Record<string, unknown>
}

export interface DatasetMeta {
  dataset_id: string
  data_path: string
  total: number
  sources: Array<{ value: string; label: string }>
  statuses: ReviewStatus[]
  issue_options: string[]
}

export interface ReviewStats {
  total: number
  reviewed: number
  progress: number
  counts: Record<ReviewStatus, number>
  issues: Record<string, number>
}

export interface ReviewUpdate {
  status: ReviewStatus
  issues: string[]
  note: string
  caption: string | null
}
