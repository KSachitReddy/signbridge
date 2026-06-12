import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { useTranslation } from 'react-i18next';

interface Log {
  emotion: string;
}

interface AnalyticsDashboardProps {
  logs: Log[];
}

export const AnalyticsDashboard = ({ logs }: AnalyticsDashboardProps) => {
  const { t } = useTranslation();
  const data = [
    { name: t('happy'), value: logs.filter((l: Log) => l.emotion === 'Happy').length },
    { name: t('neutral'), value: logs.filter((l: Log) => l.emotion === 'Neutral').length },
    { name: t('sad'), value: logs.filter((l: Log) => l.emotion === 'Sad').length },
  ];

  return (
    <div className="dashboard glass">
      <h2>{t('analyticsTitle')}</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <XAxis dataKey="name" stroke="#fff" />
          <YAxis stroke="#fff" />
          <Tooltip />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
