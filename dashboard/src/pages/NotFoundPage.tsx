import { Button, Result } from 'antd'
import { useNavigate } from 'react-router'

export default function NotFoundPage() {
  const navigate = useNavigate()
  return (
    <div className="page">
      <Result
        status="404"
        title="Страница не найдена"
        subTitle="Такого раздела в дашборде нет."
        extra={
          <Button type="primary" onClick={() => navigate('/')}>
            К оперативной обстановке
          </Button>
        }
      />
    </div>
  )
}
