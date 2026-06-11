import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export const AnalyticsDashboard = ({ logs }) => {
  // Simple aggregation for the demo
  const data = [
    { name: 'Happy', value: logs.filter((l) => l.emotion === 'Happy').length },
    { name: 'Neutral', value: logs.filter((l) => l.emotion === 'Neutral').length },
    { name: 'Sad', value: logs.filter((l) => l.emotion === 'Sad').length },
  ];

  return (
    <div className="dashboard glass">
      <h2>Analytics Dashboard</h2>
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
