import { theme, type ThemeConfig } from 'antd'

/**
 * Палитра Foresight — светлая: белые поверхности, акцент светло-голубой, цвета риска — только для риска.
 * Кнопки с белым текстом — на более насыщенном голубом (`accentStrong`), чтобы текст читался.
 * CSS-переменные той же палитры — `:root` в styles.css.
 */
export const palette = {
  bg: '#f3f6fa',
  panel: '#ffffff',
  panelRaised: '#f7fafd',
  sider: '#ffffff',
  border: '#d9e2ec',
  borderSoft: '#e7edf4',
  text: '#0f1b2d',
  muted: '#566476',
  faint: '#8a97a8',
  accent: '#0ea5e9',
  accentStrong: '#0284c7',
  accentSoft: '#e0f2fe',
  accentText: '#0369a1',
  planned: '#94a3b8',
} as const

/** Цвета ниток ТС на графике движения (не пересекаются с цветами риска; достаточно тёмные для белого фона). */
export const threadColors = [
  '#0d9488',
  '#9333ea',
  '#2563eb',
  '#db2777',
  '#7c3aed',
  '#0284c7',
  '#c026d3',
  '#4f46e5',
  '#0891b2',
  '#ea580c',
]

export const fontFamily =
  "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif"

export const antdTheme: ThemeConfig = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: palette.accent,
    colorInfo: palette.accent,
    colorLink: palette.accentStrong,
    colorSuccess: '#16a34a',
    colorWarning: '#d97706',
    colorError: '#dc2626',
    colorBgLayout: palette.bg,
    colorBgContainer: palette.panel,
    colorBorder: palette.border,
    colorBorderSecondary: palette.borderSoft,
    colorText: palette.text,
    colorTextSecondary: palette.muted,
    colorTextTertiary: palette.faint,
    borderRadius: 8,
    fontFamily,
    fontSize: 14,
    wireframe: false,
  },
  components: {
    Layout: { siderBg: palette.sider, headerBg: palette.panel, bodyBg: palette.bg, triggerBg: palette.sider },
    Menu: {
      itemBg: palette.sider,
      itemColor: palette.muted,
      itemHoverBg: '#f0f7fd',
      itemSelectedBg: palette.accentSoft,
      itemSelectedColor: palette.accentText,
      itemHeight: 44,
      iconSize: 17,
      collapsedIconSize: 18,
    },
    Button: {
      colorPrimary: palette.accentStrong,
      colorPrimaryHover: palette.accent,
      colorPrimaryActive: palette.accentText,
    },
    Tabs: {
      inkBarColor: palette.accent,
      itemSelectedColor: palette.accentStrong,
      itemHoverColor: palette.accentStrong,
    },
    Card: { headerFontSize: 14, paddingLG: 16 },
    Table: { headerBg: '#f4f8fc', rowHoverBg: '#f5faff', cellPaddingBlockSM: 6 },
    Segmented: {
      trackBg: '#edf2f7',
      itemSelectedBg: palette.panel,
      itemSelectedColor: palette.accentText,
      itemColor: palette.muted,
      itemHoverColor: palette.text,
      trackPadding: 3,
    },
  },
}
