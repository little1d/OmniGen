import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronRight,
  Circle,
  Clipboard,
  Database,
  Download,
  LoaderCircle,
  PencilLine,
  RotateCcw,
  Save,
  Search,
  X,
  XCircle,
} from 'lucide-react'

import { api } from './api'
import { MoleculeViewer } from './MoleculeViewer'
import type {
  DatasetMeta,
  MoleculeDetail,
  MoleculeSummary,
  Review,
  ReviewStats,
  ReviewStatus,
} from './types'

type DetailTab = 'caption' | 'prompt' | 'facts' | 'sdf' | 'metadata'
type StatusFilter = 'all' | ReviewStatus

const STATUS_LABELS: Record<ReviewStatus, string> = {
  pending: '待审核',
  approved: '通过',
  needs_changes: '需修改',
  rejected: '驳回',
}

const ISSUE_LABELS: Record<string, string> = {
  factual_error: '事实错误',
  unsupported_claim: '无依据断言',
  chemistry_error: '化学错误',
  geometry_error: '构象错误',
  style_issue: '表达问题',
  incomplete: '信息缺失',
  duplicate: '重复样本',
}

const STATUS_ICONS = {
  pending: Circle,
  approved: CheckCircle2,
  needs_changes: AlertTriangle,
  rejected: XCircle,
}

const PAGE_SIZE = 200

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    await navigator.clipboard.writeText(value)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 900)
  }
  return (
    <button className="icon-button" onClick={copy} title="复制" type="button">
      {copied ? <Check size={16} /> : <Clipboard size={16} />}
    </button>
  )
}

function MoleculeList({
  items,
  selectedId,
  onSelect,
}: {
  items: MoleculeSummary[]
  selectedId?: string
  onSelect: (id: string) => void
}) {
  if (!items.length) return <div className="empty-state">没有匹配的记录</div>
  return (
    <div className="molecule-list">
      {items.map((item) => {
        const StatusIcon = STATUS_ICONS[item.review.status]
        return (
          <button
            className={`molecule-row ${selectedId === item.id ? 'active' : ''}`}
            key={item.id}
            onClick={() => onSelect(item.id)}
            type="button"
          >
            <StatusIcon className={`status-icon status-${item.review.status}`} size={16} />
            <span className="molecule-row-copy">
              <span className="molecule-row-title">
                <strong>{item.display_key}</strong>
                <span>{item.iupac_name || item.source_id || item.smiles}</span>
              </span>
              <span className="molecule-row-meta">
                <span className={`source-badge source-${item.source}`}>{item.source_label}</span>
                <code>{item.formula || '–'}</code>
                <span>{item.caption_chars} 字</span>
              </span>
            </span>
            <ChevronRight className="row-arrow" size={15} />
          </button>
        )
      })}
    </div>
  )
}

function FactsView({ molecule }: { molecule: MoleculeDetail }) {
  const facts = molecule.facts ?? {}
  const shape = (facts.shape_descriptors_for_current_conformer ?? {}) as Record<string, unknown>
  const cards = [
    ['重原子', facts.heavy_atom_count],
    ['显式氢', facts.explicit_h_count],
    ['氢键供体', facts.hbond_donors],
    ['氢键受体', facts.hbond_acceptors],
    ['可旋转键', facts.rotatable_bonds],
    ['环 / 芳香环', `${facts.ring_count ?? '–'} / ${facts.aromatic_ring_count ?? '–'}`],
    ['回转半径', shape.radius_of_gyration_A ? `${shape.radius_of_gyration_A} Å` : '–'],
    ['非球面性', shape.asphericity],
  ]
  return (
    <>
      <div className="fact-grid">
        {cards.map(([label, value]) => (
          <div className="fact" key={String(label)}>
            <strong>{String(value ?? '–')}</strong>
            <span>{String(label)}</span>
          </div>
        ))}
      </div>
      <pre className="code-block">{JSON.stringify(facts, null, 2)}</pre>
    </>
  )
}

function ReviewEditor({
  molecule,
  issueOptions,
  saving,
  onSave,
}: {
  molecule: MoleculeDetail
  issueOptions: string[]
  saving: boolean
  onSave: (review: { status: ReviewStatus; issues: string[]; note: string; caption: string | null }) => void
}) {
  const [status, setStatus] = useState<ReviewStatus>(molecule.review.status)
  const [issues, setIssues] = useState<string[]>(molecule.review.issues)
  const [note, setNote] = useState(molecule.review.note)
  const [caption, setCaption] = useState(molecule.effective_caption)
  const [editingCaption, setEditingCaption] = useState(false)

  useEffect(() => {
    setStatus(molecule.review.status)
    setIssues(molecule.review.issues)
    setNote(molecule.review.note)
    setCaption(molecule.effective_caption)
    setEditingCaption(false)
  }, [molecule.id, molecule.review.updated_at])

  const toggleIssue = (issue: string) => {
    setIssues((current) =>
      current.includes(issue) ? current.filter((value) => value !== issue) : [...current, issue],
    )
  }
  const captionOverride = caption === molecule.caption ? null : caption

  return (
    <>
      <div className="detail-content">
        <div className="content-heading">
          <span>生成文本 · {caption.length} 字</span>
          <div>
            <button
              className={`icon-button ${editingCaption ? 'active' : ''}`}
              onClick={() => setEditingCaption((value) => !value)}
              title="编辑 Caption"
              type="button"
            >
              <PencilLine size={16} />
            </button>
            <CopyButton value={caption} />
          </div>
        </div>
        {editingCaption ? (
          <textarea
            className="caption-editor"
            onChange={(event) => setCaption(event.target.value)}
            spellCheck={false}
            value={caption}
          />
        ) : (
          <article className="caption-text">{caption}</article>
        )}
      </div>
      <div className="review-editor">
        <div className="review-heading">
          <div>
            <span className="eyebrow">审核结论</span>
            <strong>{STATUS_LABELS[status]}</strong>
          </div>
          {molecule.review.updated_at && (
            <time>{new Date(molecule.review.updated_at).toLocaleString('zh-CN')}</time>
          )}
        </div>
        <div className="review-actions">
          <button
            className={status === 'approved' ? 'decision approved active' : 'decision approved'}
            onClick={() => setStatus('approved')}
            type="button"
          >
            <Check size={17} />通过
          </button>
          <button
            className={status === 'needs_changes' ? 'decision needs-changes active' : 'decision needs-changes'}
            onClick={() => setStatus('needs_changes')}
            type="button"
          >
            <PencilLine size={17} />需修改
          </button>
          <button
            className={status === 'rejected' ? 'decision rejected active' : 'decision rejected'}
            onClick={() => setStatus('rejected')}
            type="button"
          >
            <X size={17} />驳回
          </button>
          <button
            className="icon-button"
            onClick={() => {
              setStatus('pending')
              setIssues([])
              setNote('')
            }}
            title="清除审核"
            type="button"
          >
            <RotateCcw size={16} />
          </button>
        </div>
        <div className="issue-grid">
          {issueOptions.map((issue) => (
            <label className={issues.includes(issue) ? 'issue-chip checked' : 'issue-chip'} key={issue}>
              <input checked={issues.includes(issue)} onChange={() => toggleIssue(issue)} type="checkbox" />
              {ISSUE_LABELS[issue] ?? issue}
            </label>
          ))}
        </div>
        <textarea
          className="review-note"
          onChange={(event) => setNote(event.target.value)}
          placeholder="审核备注"
          value={note}
        />
        <button
          className="save-button"
          disabled={saving}
          onClick={() => onSave({ status, issues, note, caption: captionOverride })}
          type="button"
        >
          {saving ? <LoaderCircle className="spin" size={17} /> : <Save size={17} />}
          保存并审核下一条
        </button>
      </div>
    </>
  )
}

export default function App() {
  const [meta, setMeta] = useState<DatasetMeta | null>(null)
  const [stats, setStats] = useState<ReviewStats | null>(null)
  const [items, setItems] = useState<MoleculeSummary[]>([])
  const [total, setTotal] = useState(0)
  const [current, setCurrent] = useState<MoleculeDetail | null>(null)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [source, setSource] = useState('all')
  const [status, setStatus] = useState<StatusFilter>('all')
  const [tab, setTab] = useState<DetailTab>('caption')
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const loadStats = useCallback(async () => setStats(await api.stats()), [])

  const fetchItems = useCallback(async (offset = 0) => {
    const params = new URLSearchParams({
      limit: String(PAGE_SIZE),
      offset: String(offset),
    })
    if (debouncedQuery) params.set('q', debouncedQuery)
    if (source !== 'all') params.set('source', source)
    if (status !== 'all') params.set('status', status)
    return api.items(params)
  }, [debouncedQuery, source, status])

  const loadItems = useCallback(async (offset = 0, append = false) => {
    const result = await fetchItems(offset)
    setItems((currentItems) => (append ? [...currentItems, ...result.items] : result.items))
    setTotal(result.total)
    return result.items
  }, [fetchItems])

  const loadMore = async () => {
    setLoadingMore(true)
    try {
      await loadItems(items.length, true)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法加载更多记录')
    } finally {
      setLoadingMore(false)
    }
  }

  const selectItem = useCallback(async (id: string) => {
    setError('')
    try {
      setCurrent(await api.item(id))
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '无法读取记录')
    }
  }, [])

  useEffect(() => {
    const timeout = window.setTimeout(() => setDebouncedQuery(query), 180)
    return () => window.clearTimeout(timeout)
  }, [query])

  useEffect(() => {
    setLoading(true)
    Promise.all([api.meta(), api.stats(), loadItems()])
      .then(([metaResult, statsResult, itemResults]) => {
        setMeta(metaResult)
        setStats(statsResult)
        if (itemResults[0]) selectItem(itemResults[0].id)
      })
      .catch((requestError) =>
        setError(requestError instanceof Error ? requestError.message : '无法加载数据集'),
      )
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!meta) return
    loadItems()
      .then((nextItems) => {
        if (!nextItems.length) {
          setCurrent(null)
        } else if (!current || !nextItems.some((item) => item.id === current.id)) {
          selectItem(nextItems[0].id)
        }
      })
      .catch((requestError) =>
        setError(requestError instanceof Error ? requestError.message : '无法筛选数据'),
      )
  }, [debouncedQuery, source, status])

  const currentIndex = useMemo(
    () => items.findIndex((item) => item.id === current?.id),
    [items, current?.id],
  )

  const moveSelection = useCallback(
    (direction: number) => {
      if (!items.length) return
      const nextIndex = (Math.max(currentIndex, 0) + direction + items.length) % items.length
      selectItem(items[nextIndex].id)
    },
    [currentIndex, items, selectItem],
  )

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return
      if (event.key === 'ArrowDown' || event.key === 'ArrowRight') moveSelection(1)
      if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') moveSelection(-1)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [moveSelection])

  const saveReview = async (review: {
    status: ReviewStatus
    issues: string[]
    note: string
    caption: string | null
  }) => {
    if (!current) return
    setSaving(true)
    setError('')
    try {
      const saved: Review = await api.saveReview(current.id, review)
      const effectiveCaption = saved.caption ?? current.caption
      const updatedItems = items.map((item) =>
        item.id === current.id
          ? { ...item, review: saved, caption_chars: effectiveCaption.length }
          : item,
      )
      const removed = status !== 'all' && saved.status !== status
      let visibleItems = removed
        ? updatedItems.filter((item) => item.id !== current.id)
        : updatedItems
      let visibleTotal = removed ? Math.max(0, total - 1) : total
      let nextIndex = Math.max(removed ? currentIndex : currentIndex + 1, 0)

      setItems(visibleItems)
      setTotal(visibleTotal)
      await loadStats()

      if (nextIndex >= visibleItems.length && visibleItems.length < visibleTotal) {
        const nextPage = await fetchItems(visibleItems.length)
        visibleItems = [...visibleItems, ...nextPage.items]
        visibleTotal = nextPage.total
        setItems(visibleItems)
        setTotal(visibleTotal)
      }

      if (visibleItems.length) {
        if (nextIndex >= visibleItems.length) nextIndex = 0
        await selectItem(visibleItems[nextIndex].id)
      } else {
        setCurrent(null)
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const detailBody = () => {
    if (!current) return <div className="empty-state">请选择一条记录</div>
    if (tab === 'caption') {
      return (
        <ReviewEditor
          issueOptions={meta?.issue_options ?? []}
          molecule={current}
          onSave={saveReview}
          saving={saving}
        />
      )
    }
    const values: Record<Exclude<DetailTab, 'caption' | 'facts' | 'metadata'>, string> = {
      prompt: current.prompt || '当前记录没有 Prompt',
      sdf: current.sdf || '当前记录没有 SDF',
    }
    return (
      <div className="detail-content standalone">
        <div className="content-heading">
          <span>
            {tab === 'prompt' ? current.prompt_label || 'Prompt' : tab === 'sdf' ? 'SDF' : '完整元数据'}
          </span>
          {tab !== 'facts' && <CopyButton value={tab === 'metadata' ? metadataText : values[tab]} />}
        </div>
        {tab === 'facts' ? (
          <FactsView molecule={current} />
        ) : (
          <pre className="code-block">{tab === 'metadata' ? metadataText : values[tab]}</pre>
        )}
      </div>
    )
  }

  const metadataText = current
    ? JSON.stringify(
        {
          source: current.source_label,
          source_id: current.source_id,
          smiles: current.smiles,
          iupac_name: current.iupac_name,
          formula: current.formula,
          model: current.model,
          usage: current.usage,
          provenance: current.provenance,
        },
        null,
        2,
      )
    : ''

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">O</span>
          <div>
            <h1>OmniGen Review</h1>
            <p>{meta ? `${meta.dataset_id} · ${meta.total} 条记录` : '本地数据质量审核'}</p>
          </div>
        </div>
        <label className="search-box">
          <Search size={17} />
          <input
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索名称、SMILES、分子式或来源"
            value={query}
          />
        </label>
        <label className="source-select">
          <Database size={16} />
          <select onChange={(event) => setSource(event.target.value)} value={source}>
            <option value="all">全部来源</option>
            {meta?.sources.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <a className="icon-button" href="/api/export" title="导出审核结果">
          <Download size={17} />
        </a>
        <div className="progress-summary">
          <div>
            <strong>{stats?.reviewed ?? 0}</strong>
            <span>/ {stats?.total ?? 0} 已审核</span>
          </div>
          <div className="progress-track">
            <span style={{ width: `${(stats?.progress ?? 0) * 100}%` }} />
          </div>
        </div>
      </header>

      <main className="workspace">
        <aside className="queue-panel">
          <div className="queue-header">
            <div>
              <span className="eyebrow">审核队列</span>
              <strong>{total} 条</strong>
            </div>
            <div className="status-filter" aria-label="审核状态筛选">
              {(['all', 'pending', 'approved', 'needs_changes', 'rejected'] as StatusFilter[]).map(
                (value) => {
                  const StatusIcon = value === 'all' ? null : STATUS_ICONS[value]
                  return (
                    <button
                      className={status === value ? 'active' : ''}
                      key={value}
                      onClick={() => setStatus(value)}
                      title={value === 'all' ? '全部' : STATUS_LABELS[value]}
                      type="button"
                    >
                      {StatusIcon && <StatusIcon size={12} />}
                      {value === 'all' ? '全部' : stats?.counts[value] ?? 0}
                    </button>
                  )
                },
              )}
            </div>
          </div>
          <MoleculeList items={items} onSelect={selectItem} selectedId={current?.id} />
          {items.length < total && (
            <button
              className="load-more-button"
              disabled={loadingMore}
              onClick={loadMore}
              type="button"
            >
              {loadingMore ? <LoaderCircle className="spin" size={14} /> : null}
              加载更多 · {items.length}/{total}
            </button>
          )}
        </aside>

        {current ? (
          <MoleculeViewer molecule={current} />
        ) : (
          <section className="viewer-panel empty-state">{loading ? '正在加载…' : '没有可显示的分子'}</section>
        )}

        <section className="detail-panel">
          <nav className="detail-tabs">
            {(['caption', 'prompt', 'facts', 'sdf', 'metadata'] as DetailTab[]).map((value) => (
              <button
                className={tab === value ? 'active' : ''}
                key={value}
                onClick={() => setTab(value)}
                type="button"
              >
                {{ caption: 'Caption', prompt: 'Prompt', facts: '结构事实', sdf: 'SDF', metadata: '元数据' }[
                  value
                ]}
              </button>
            ))}
          </nav>
          {detailBody()}
        </section>
      </main>
      {error && (
        <div className="error-toast" role="alert">
          <XCircle size={17} />
          {error}
        </div>
      )}
    </div>
  )
}
