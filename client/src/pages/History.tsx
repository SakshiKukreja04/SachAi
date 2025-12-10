import { ParticleBackground } from '@/components/ParticleBackground';
import { Navbar } from '@/components/Navbar';
import { HistoryCard } from '@/components/HistoryCard';
import { GlassCard } from '@/components/GlassCard';
import { History as HistoryIcon, Search } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';

// Demo data
const demoHistory = [
  {
    id: 'demo-123',
    date: 'Dec 10, 2025 • 2:34 PM',
    score: 87,
    label: 'deepfake' as const,
    thumbnail: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=200&h=150&fit=crop',
  },
  {
    id: 'demo-124',
    date: 'Dec 9, 2025 • 10:15 AM',
    score: 23,
    label: 'authentic' as const,
    thumbnail: 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200&h=150&fit=crop',
  },
  {
    id: 'demo-125',
    date: 'Dec 8, 2025 • 4:45 PM',
    score: 56,
    label: 'suspected' as const,
    thumbnail: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=200&h=150&fit=crop',
  },
  {
    id: 'demo-126',
    date: 'Dec 7, 2025 • 9:20 AM',
    score: 12,
    label: 'authentic' as const,
    thumbnail: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200&h=150&fit=crop',
  },
  {
    id: 'demo-127',
    date: 'Dec 6, 2025 • 3:10 PM',
    score: 92,
    label: 'deepfake' as const,
    thumbnail: 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=200&h=150&fit=crop',
  },
];

const History = () => {
  const [search, setSearch] = useState('');
  
  const filteredHistory = demoHistory.filter((item) =>
    item.date.toLowerCase().includes(search.toLowerCase()) ||
    item.label.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-hero-gradient relative overflow-hidden">
      <ParticleBackground />
      <Navbar />
      
      <main className="relative z-10 pt-24 pb-16">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto">
            {/* Header */}
            <div className="text-center mb-8 animate-fade-in">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-secondary/20 flex items-center justify-center mx-auto mb-4">
                <HistoryIcon className="w-8 h-8 text-primary" />
              </div>
              <h1 className="font-heading text-3xl md:text-4xl font-bold mb-2">Analysis History</h1>
              <p className="text-muted-foreground">View your previous video analyses</p>
            </div>

            {/* Search */}
            <div className="relative mb-6 animate-fade-in-up">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by date or status..."
                className={cn(
                  'w-full h-12 pl-12 pr-4 rounded-lg',
                  'glass border-white/20',
                  'text-foreground placeholder:text-muted-foreground',
                  'focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent',
                  'transition-all duration-300'
                )}
              />
            </div>

            {/* History List */}
            {filteredHistory.length > 0 ? (
              <div className="space-y-4">
                {filteredHistory.map((item, index) => (
                  <HistoryCard
                    key={item.id}
                    {...item}
                    delay={index * 100}
                  />
                ))}
              </div>
            ) : (
              <GlassCard className="text-center py-12">
                <p className="text-muted-foreground">No analyses found matching your search.</p>
              </GlassCard>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default History;
