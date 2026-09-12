import { useEffect, useRef, useState } from 'react'
import * as $3Dmol from '3dmol'
import { Download, Eye, EyeOff, Image, Pause, Play, RotateCcw } from 'lucide-react'

import type { MoleculeDetail } from './types'

type ViewStyle = 'stick' | 'line' | 'sphere'

interface MoleculeViewerProps {
  molecule: MoleculeDetail
}

export function MoleculeViewer({ molecule }: MoleculeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<ReturnType<typeof $3Dmol.createViewer> | null>(null)
  const [style, setStyle] = useState<ViewStyle>('stick')
  const [showHydrogen, setShowHydrogen] = useState(true)
  const [spinning, setSpinning] = useState(false)
  const [show2D, setShow2D] = useState(false)
  const [renderError, setRenderError] = useState('')

  const applyStyle = () => {
    const viewer = viewerRef.current
    if (!viewer) return
    const molecularStyle =
      style === 'line'
        ? { line: { colorscheme: 'Jmol', linewidth: 2 } }
        : style === 'sphere'
          ? { sphere: { colorscheme: 'Jmol', scale: 0.82 } }
          : {
              stick: { colorscheme: 'Jmol', radius: 0.17 },
              sphere: { colorscheme: 'Jmol', scale: 0.27 },
            }
    viewer.setStyle({}, {})
    viewer.setStyle({}, molecularStyle as never)
    if (!showHydrogen) viewer.setStyle({ elem: 'H' }, {})
    viewer.render()
  }

  useEffect(() => {
    if (!containerRef.current) return
    if (!viewerRef.current) {
      viewerRef.current = $3Dmol.createViewer(containerRef.current, {
        backgroundColor: '#102722',
        antialias: true,
      })
    }
    const viewer = viewerRef.current
    viewer.clear()
    setRenderError('')
    try {
      if (!molecule.sdf) throw new Error('当前记录没有 SDF')
      viewer.addModel(molecule.sdf, 'sdf')
      applyStyle()
      viewer.zoomTo()
      viewer.render()
      if (spinning) viewer.spin('y', 1)
    } catch (error) {
      setRenderError(error instanceof Error ? error.message : '无法渲染当前分子')
    }
    setShow2D(false)
  }, [molecule.id])

  useEffect(applyStyle, [style, showHydrogen])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return
    viewer.spin(spinning ? 'y' : false, 1)
  }, [spinning])

  useEffect(() => {
    const element = containerRef.current
    if (!element) return
    const observer = new ResizeObserver(() => viewerRef.current?.resize())
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  const download = () => {
    if (!molecule.sdf) return
    const url = URL.createObjectURL(new Blob([molecule.sdf], { type: 'chemical/x-mdl-sdfile' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${molecule.display_key}.sdf`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const reset = () => {
    viewerRef.current?.zoomTo()
    viewerRef.current?.render()
  }

  return (
    <section className="viewer-panel">
      <header className="molecule-identity">
        <div className="identity-line">
          <span className={`source-badge source-${molecule.source}`}>{molecule.source_label}</span>
          <h2>{molecule.display_key}</h2>
          <code>{molecule.formula || 'formula unavailable'}</code>
        </div>
        <p>{molecule.iupac_name || molecule.source_id || molecule.smiles}</p>
      </header>

      <div className="molecule-stage">
        <div ref={containerRef} className={show2D ? 'viewer-3d is-hidden' : 'viewer-3d'} />
        {show2D && molecule.png && <img className="viewer-2d" src={molecule.png} alt={molecule.display_key} />}
        {renderError && <div className="viewer-error">{renderError}</div>}
      </div>

      <footer className="viewer-toolbar">
        <div className="segmented" aria-label="分子显示模式">
          {(['stick', 'line', 'sphere'] as ViewStyle[]).map((value) => (
            <button
              className={style === value ? 'active' : ''}
              key={value}
              onClick={() => setStyle(value)}
              type="button"
            >
              {{ stick: '球棍', line: '线框', sphere: '空间填充' }[value]}
            </button>
          ))}
        </div>
        <button
          className={`icon-button ${showHydrogen ? 'active' : ''}`}
          onClick={() => setShowHydrogen((value) => !value)}
          title={showHydrogen ? '隐藏氢原子' : '显示氢原子'}
          type="button"
        >
          {showHydrogen ? <Eye size={16} /> : <EyeOff size={16} />}
        </button>
        <button
          className={`icon-button ${spinning ? 'active' : ''}`}
          onClick={() => setSpinning((value) => !value)}
          title={spinning ? '停止旋转' : '自动旋转'}
          type="button"
        >
          {spinning ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button
          className={`icon-button ${show2D ? 'active' : ''}`}
          disabled={!molecule.png}
          onClick={() => setShow2D((value) => !value)}
          title={molecule.png ? '切换 2D/3D' : '当前记录没有 2D 图'}
          type="button"
        >
          <Image size={16} />
        </button>
        <span className="toolbar-spacer" />
        <button className="icon-button" onClick={reset} title="重置视角" type="button">
          <RotateCcw size={16} />
        </button>
        <button className="icon-button" disabled={!molecule.sdf} onClick={download} title="下载 SDF" type="button">
          <Download size={16} />
        </button>
      </footer>
    </section>
  )
}
