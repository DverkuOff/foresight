import { Spin } from 'antd'

export function PageSpinner() {
  return (
    <div style={{ height: 'calc(100vh - 52px)', display: 'grid', placeItems: 'center' }}>
      <Spin size="large" />
    </div>
  )
}
